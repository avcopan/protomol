#!/usr/bin/env python
"""Stress-test protomol's canonical form on RGD1 and stjohn-radicals.

Run it in protomol's dev environment (it has protomol, autogeom and RDKit):

    pixi run -e dev python benchmarks/canonical_stress.py run --sample    # 1 %, 100 relabellings
    pixi run -e dev python benchmarks/canonical_stress.py run --workers 4 # everything, 5 relabellings
    pixi run -e dev python benchmarks/canonical_stress.py summary --sample
    pixi run -e dev python benchmarks/canonical_stress.py parquet --sample

Stage 1 is `extract_rgd1_geometries.py` (the Gaussian plugin lives in another env); run it
first, with or without `--sample`, to make the RGD1 geometries.

Species: every RGD1 `molN` in `RandP_smiles.txt` (SMILES; geometry from stage 1, when its
log parsed and has NImag = 0) and every record of the stjohn SDF (SMILES field and
coordinates, charge and spin from the SMILES). Nothing is unpacked.

Per species (a time cap of `--cap` seconds on each phase, every exception recorded):
  1. `canonical_json` unchanged under `--relabels` random relabellings (the SMILES graph,
     and the geometry graph when there is one);
  2. `from_smiles` and `from_geometry` agree (full canonical JSON when the SMILES
     determines all stereo, else the stereo-free one), with a category when they don't;
  3. the data for the duplicate check (hashes of the canonical JSON, standard InChI);
  4. the data for the InChI partition comparison (stereo-free canonical form vs InChI
     without stereo, full vs standard InChI), plus flags for categorizing disagreements;
  5. the time of each step, against the heavy-atom count.

The rows go to a JSON-lines file as they finish, so a run is resumable (a rerun skips the
ids in the file); `parquet` converts it, `summary` prints the counts. The pool leaves two
cores free (at most `cores - 2` workers).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "code/am/data"))
RGD1_ZIP = DATA / "rgd1/RGD1_rawoutput.zip"
STJOHN_SDF = DATA / "stjohn-radicals/20200415_radical_database.sdf.gz"
SWEEPS = DATA / "sweeps"


def in_sample(species_id: str) -> bool:
    """The 1 % sample: a stable hash of the id (the same rule as the extractor)."""
    return hashlib.sha1(species_id.encode()).digest()[0] % 100 == 0


def sha(text: str) -> str:
    """Short digest of a canonical JSON."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- inputs


def rgd1_species(sample: bool) -> Iterator[dict[str, Any]]:
    """RGD1 species (id, smiles, geometry or the reason it has none)."""
    geos: dict[str, dict[str, Any]] = {}
    path = SWEEPS / f"rgd1-rp-geometries{'-sample' if sample else ''}.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt") as handle:
            for line in handle:
                rec = json.loads(line)
                geos[rec["id"]] = rec
    else:
        print(
            f"warning: {path} not found; run extract_rgd1_geometries.py",
            file=sys.stderr,
        )
    with zipfile.ZipFile(RGD1_ZIP) as archive:
        lines = archive.read("raw_output/RandP_smiles.txt").decode().splitlines()[1:]
    for line in lines:
        if not line.strip():
            continue
        smiles, mol = line.split()
        if sample and not in_sample(f"rgd1:{mol}"):
            continue
        geo = geos.get(mol, {"status": "skipped: no extraction"})
        yield {
            "id": f"rgd1:{mol}",
            "dataset": "rgd1",
            "smiles": smiles,
            "geo_status": geo["status"],
            "symbols": geo.get("symbols"),
            "coords": geo.get("coords"),
        }


def stjohn_species(sample: bool) -> Iterator[dict[str, Any]]:
    """Stjohn species: the SDF records, as text (parsed in the workers)."""
    block: list[str] = []
    with gzip.open(STJOHN_SDF, "rt") as handle:
        for line in handle:
            if line.startswith("$$$$"):
                title = block[0].strip()
                if not sample or in_sample(f"stjohn:{title}"):
                    yield {
                        "id": f"stjohn:{title}",
                        "dataset": "stjohn",
                        "sdf": "".join(block),
                    }
                block = []
            else:
                block.append(line)


def parse_sdf(sdf: str) -> dict[str, Any]:
    """SMILES, symbols and coordinates of one SDF record."""
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(
        sdf.split("M  END", maxsplit=1)[0] + "M  END\n", sanitize=False, removeHs=False
    )
    if mol is None:
        return {"geo_status": "error: unreadable SDF"}
    smiles = sdf.split(">  <SMILES>")[1].splitlines()[1].strip()
    xyz = mol.GetConformer().GetPositions()
    return {
        "smiles": smiles,
        "geo_status": "ok",
        "symbols": [a.GetSymbol() for a in mol.GetAtoms()],
        "coords": [[round(float(x), 6) for x in row] for row in xyz],
    }


# ------------------------------------------------------------------------------ checks


class Cap(Exception):
    """A phase ran over its time cap."""


@contextlib.contextmanager
def capped(seconds: int) -> Iterator[None]:
    """Raise Cap after `seconds` (SIGALRM; native code only stops when it returns)."""

    def handler(signum: int, frame: object) -> None:
        raise Cap

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def fail(exc: BaseException) -> str:
    """One-line description of an exception ('timeout' for a Cap)."""
    if isinstance(exc, Cap):
        return "timeout"
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"


def relabel_check(gra: Any, n: int, seed: str) -> tuple[int, str | None]:
    """Failures of canonical_json under `n` random relabellings (and the first seed)."""
    import protomol as pm

    ref = pm.canonical_json(gra)
    rng = random.Random(seed)
    bad, first = 0, None
    for _ in range(n):
        perm = list(range(len(gra.symbols)))
        rng.shuffle(perm)
        if pm.canonical_json(pm.relabel(gra, perm)) != ref:
            bad += 1
            first = first or json.dumps(perm)
    return bad, first


def inchis(smiles: str) -> dict[str, Any]:
    """Standard InChI, InChI without stereo, and structural flags, from the SMILES."""
    from rdkit import Chem
    from rdkit.Chem import inchi, rdCIPLabeler

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"inchi_error": "RDKit cannot read the SMILES"}
    out: dict[str, Any] = {
        "inchi": inchi.MolToInchi(mol),
        "inchi_nostereo": inchi.MolToInchi(mol, options="-SNon"),
    }
    out["n_heavy"] = mol.GetNumHeavyAtoms()
    out["is_radical"] = any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    out["is_charged"] = any(a.GetFormalCharge() for a in mol.GetAtoms())
    out["has_amine_n"] = mol.HasSubstructMatch(
        Chem.MolFromSmarts(
            "[NX3;!$(N-[#6,#7,#8]=,:*);!$(N=*);!$(N#*)]([#6,#1,#7,#8])([#6,#1,#7,#8])[#6,#1,#7,#8]"
        )
    )
    out["has_chain"] = mol.HasSubstructMatch(Chem.MolFromSmarts("[*]=[*;D2]=[*]"))
    out["has_conjugated"] = any(b.GetIsConjugated() for b in mol.GetBonds())
    out["has_pseudo"] = False
    with contextlib.suppress(Exception):
        rdCIPLabeler.AssignCIPLabels(mol)
        out["has_pseudo"] = any(
            a.HasProp("_CIPCode") and a.GetProp("_CIPCode") in ("r", "s")
            for a in mol.GetAtoms()
        )
    return out


def classify(gs: Any, gg: Any, determined: bool) -> str:
    """Why `from_smiles` and `from_geometry` differ: the first matching category."""
    import protomol as pm

    if (gs.charge, gs.spin) != (gg.charge, gg.spin):
        return "charge_spin"
    if pm.canonical_json(pm.without_stereo(gs)) != pm.canonical_json(
        pm.without_stereo(gg)
    ):
        return "connectivity"
    return "stereo"


def check_species(spec: dict[str, Any], relabels: int, cap: int) -> dict[str, Any]:
    """Run every check on one species and return its row."""
    import autogeom

    import protomol as pm

    warnings.simplefilter("ignore")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    row: dict[str, Any] = {"id": spec["id"], "dataset": spec["dataset"]}
    if spec["dataset"] == "stjohn":
        spec = {**spec, **parse_sdf(spec["sdf"])}
    row["smiles"] = spec.get("smiles")
    row["geo_status"] = spec.get("geo_status")
    if not spec.get("smiles"):
        row["smiles_error"] = spec["geo_status"]
        return row
    with contextlib.suppress(Exception):
        row.update(inchis(spec["smiles"]))

    gs = None
    t = time.perf_counter()
    try:
        with capped(cap):
            gs = pm.from_smiles(spec["smiles"])
        row["n_atoms"] = len(gs.symbols)
        row["charge"], row["spin"] = gs.charge, gs.spin
        row["smiles_determined"] = pm.is_determined(gs)
    except BaseException as exc:  # noqa: BLE001
        row["smiles_error"] = fail(exc)
    row["t_from_smiles"] = time.perf_counter() - t

    cj_s = cj_s0 = None
    if gs is not None:
        t = time.perf_counter()
        try:
            with capped(cap):
                cj_s = pm.canonical_json(gs)
                row["cj_hash"] = sha(cj_s)
                cj_s0 = pm.canonical_json(pm.without_stereo(gs))
                row["cj0_hash"] = sha(cj_s0)
        except BaseException as exc:  # noqa: BLE001
            row["canon_error"] = fail(exc)
        row["t_canon"] = time.perf_counter() - t
        t = time.perf_counter()
        try:
            with capped(cap * 4):
                row["relabel_smiles_bad"], row["relabel_smiles_perm"] = relabel_check(
                    gs, relabels, spec["id"]
                )
        except BaseException as exc:  # noqa: BLE001
            row["relabel_smiles_error"] = fail(exc)
        row["t_relabel"] = time.perf_counter() - t

    if spec.get("symbols") is not None and gs is not None:
        gg = None
        t = time.perf_counter()
        try:
            with capped(cap):
                geo = autogeom.Geometry(
                    symbols=spec["symbols"],
                    coordinates=spec["coords"],
                    charge=gs.charge,
                    spin=gs.spin,
                )
                gg = pm.from_geometry(geo)
        except BaseException as exc:  # noqa: BLE001
            row["geo_error"] = fail(exc)
            row["geo_category"] = (
                "timeout"
                if isinstance(exc, Cap)
                else "no_lewis"
                if isinstance(exc, pm.GraphError)
                else "error"
            )
        row["t_from_geometry"] = time.perf_counter() - t
        if gg is not None:
            t = time.perf_counter()
            try:
                with capped(cap):
                    cj_g = pm.canonical_json(gg)
                    row["cj_geo_hash"] = sha(cj_g)
                    row["cj0_geo_hash"] = sha(pm.canonical_json(pm.without_stereo(gg)))
                    determined = bool(row.get("smiles_determined"))
                    if determined:
                        same = cj_g == cj_s
                    else:
                        same = row["cj0_geo_hash"] == row.get("cj0_hash")
                    row["geo_compared"] = "full" if determined else "stereo_free"
                    row["geo_match"] = same
                    if same:
                        row["geo_category"] = (
                            "match" if determined else "match (SMILES undetermined)"
                        )
                    else:
                        cat = classify(gs, gg, determined)
                        row["geo_category"] = (
                            cat if determined else f"{cat} (SMILES undetermined)"
                        )
                        row["cj_smiles"], row["cj_geo"] = cj_s, cj_g
                        row["geo_symbols"], row["geo_coords"] = (
                            spec["symbols"],
                            spec["coords"],
                        )
            except BaseException as exc:  # noqa: BLE001
                row["geo_error"] = fail(exc)
                row["geo_category"] = "timeout" if isinstance(exc, Cap) else "error"
            row["t_canon_geo"] = time.perf_counter() - t
            t = time.perf_counter()
            try:
                with capped(cap * 4):
                    row["relabel_geo_bad"], row["relabel_geo_perm"] = relabel_check(
                        gg, relabels, spec["id"] + "/geo"
                    )
                    if row["relabel_geo_bad"]:
                        row["geo_symbols"], row["geo_coords"] = (
                            spec["symbols"],
                            spec["coords"],
                        )
            except BaseException as exc:  # noqa: BLE001
                row["relabel_geo_error"] = fail(exc)
            row["t_relabel_geo"] = time.perf_counter() - t
    elif spec.get("symbols") is None:
        row["geo_category"] = f"no geometry ({spec.get('geo_status')})"
    return row


def work(args: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    """Pool entry point: never raises."""
    spec, relabels, cap = args
    try:
        return check_species(spec, relabels, cap)
    except BaseException:  # noqa: BLE001
        return {
            "id": spec["id"],
            "dataset": spec["dataset"],
            "harness_error": traceback.format_exc()[-500:],
        }


# ------------------------------------------------------------------------------ driver


def results_path(args: argparse.Namespace) -> Path:
    """The JSON-lines file of a run."""
    if args.out:
        return args.out
    day = date.today()
    return SWEEPS / f"canonical-{'sample-' if args.sample else ''}{day}.jsonl"


def run(args: argparse.Namespace) -> None:
    """Submit every species to the pool; append rows as they finish."""
    out = results_path(args)
    relabels = args.relabels or (100 if args.sample else 5)
    workers = min(args.workers, max(1, (os.cpu_count() or 3) - 2))
    done: set[str] = set()
    if out.exists():
        with out.open() as handle:
            done = {json.loads(line)["id"] for line in handle if line.strip()}

    def species() -> Iterator[dict[str, Any]]:
        if args.dataset in ("rgd1", "all"):
            yield from rgd1_species(args.sample)
        if args.dataset in ("stjohn", "all"):
            yield from stjohn_species(args.sample)

    started, count = time.perf_counter(), 0
    in_flight = 4 * workers
    with out.open("a") as handle, ProcessPoolExecutor(workers) as pool:
        pending = set()

        def drain() -> None:
            nonlocal pending, count
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                handle.write(json.dumps(future.result(), default=str) + "\n")
                count += 1
            handle.flush()
            if count % 1000 < len(finished):
                rate = count / (time.perf_counter() - started)
                print(f"{count} done ({rate:.1f}/s)", file=sys.stderr, flush=True)

        for spec in species():
            if spec["id"] in done:
                continue
            if args.limit and count + len(pending) >= args.limit:
                break
            pending.add(pool.submit(work, (spec, relabels, args.cap)))
            while len(pending) >= in_flight:
                drain()
        while pending:
            drain()
    print(
        f"{count} species ({len(done)} already done), {relabels} relabellings, {workers} workers, "
        f"{time.perf_counter() - started:.0f} s -> {out}"
    )


def load(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The rows of a results file."""
    with results_path(args).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(args: argparse.Namespace) -> None:
    """Counts per check and category (the numbers of the report)."""
    rows = load(args)
    print(f"{len(rows)} species: {dict(Counter(r['dataset'] for r in rows))}")
    print("\nharness errors:", sum("harness_error" in r for r in rows))
    print(
        "from_smiles errors:",
        Counter(
            r["smiles_error"][:60] for r in rows if "smiles_error" in r
        ).most_common(8),
    )
    print(
        "canonical_json errors:",
        Counter(r["canon_error"][:60] for r in rows if "canon_error" in r).most_common(
            8
        ),
    )
    print("\n1. relabelling failures:")
    for key in ("smiles", "geo"):
        n = sum(1 for r in rows if f"relabel_{key}_bad" in r)
        bad = [r for r in rows if r.get(f"relabel_{key}_bad")]
        err = [r for r in rows if f"relabel_{key}_error" in r]
        print(
            f"  {key}: {n} checked, {len(bad)} species with a failure, {len(err)} errors/timeouts"
        )
    print(
        "\n2. from_smiles vs from_geometry:",
        Counter(r.get("geo_category", "-") for r in rows).most_common(),
    )
    print("\n3. duplicates by InChI:")
    by_inchi: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if "inchi" in r and "cj_hash" in r:
            by_inchi[r["inchi"]].add(r["cj_hash"])
    groups = list(by_inchi.values())
    print(
        f"  {len(groups)} InChI classes, {sum(len(v) > 1 for v in groups)} with more than one canonical JSON"
    )
    print("\n4. partitions (InChI vs canonical form):")
    for label, ikey, ckey in (
        ("stereo-free", "inchi_nostereo", "cj0_hash"),
        ("full", "inchi", "cj_hash"),
    ):
        i2c: dict[str, set[str]] = defaultdict(set)
        c2i: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            if ikey in r and ckey in r:
                i2c[r[ikey]].add(r[ckey])
                c2i[r[ckey]].add(r[ikey])
        print(
            f"  {label}: InChI classes split by canonical form: {sum(len(v) > 1 for v in i2c.values())}; "
            f"canonical classes split by InChI: {sum(len(v) > 1 for v in c2i.values())}"
        )
    print("\n5. slowest 5 (canonical_json of the SMILES graph):")
    for r in sorted((r for r in rows if "t_canon" in r), key=lambda r: -r["t_canon"])[
        :5
    ]:
        print(
            f"  {r['id']} heavy={r.get('n_heavy')} {r['t_canon']:.2f} s {r['smiles']}"
        )


def parquet(args: argparse.Namespace) -> None:
    """Convert the results to parquet (needs pyarrow, which protomol's env lacks)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = load(args)
    keys = sorted({k for r in rows for k in r})
    table = pa.table({k: [r.get(k) for r in rows] for k in keys})
    path = results_path(args).with_suffix(".parquet")
    pq.write_table(table, path)
    print(f"{len(rows)} rows -> {path}")


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("run", run), ("summary", summary), ("parquet", parquet)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument(
            "--sample", action="store_true", help="the 1 %% sample (100 relabellings)"
        )
        p.add_argument("--out", type=Path, help="results file (JSON lines)")
        if name == "run":
            p.add_argument(
                "--dataset", choices=["rgd1", "stjohn", "all"], default="all"
            )
            p.add_argument("--workers", type=int, default=4)
            p.add_argument(
                "--relabels", type=int, help="default: 100 on the sample, else 5"
            )
            p.add_argument("--cap", type=int, default=30, help="seconds per phase")
            p.add_argument("--limit", type=int, help="stop after about N species")
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
