#!/usr/bin/env python
"""Write the canonical stress-test report and its reproducers from a results file.

Reads the JSON lines of `canonical_stress.py run` (no pyarrow needed) and writes

- the report (counts per check and category, with examples), and
- one standalone reproducer script per distinct failure, `<report stem>/repro/*.py`
  (exit status 1 = the failure reproduces; each is run here and the result is listed),
- `<report stem>/slowest-100.csv`.

Run it in protomol's dev env, again at any time (also while `run` is still going: it only
reads the file):

    pixi run -e dev python benchmarks/canonical_report.py \
        --results ~/code/am/data/sweeps/canonical-2026-11-24.jsonl \
        --report ~/proj/pm/reports/2026-11-canonical-stress-test.md

Everything below the line `<!-- hand-written -->` in an existing report is kept when the
report is regenerated; that is where the "For Maxim" section and the notes live.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MARKER = "<!-- hand-written -->"
ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict[str, Any]]:
    """The rows of a results file."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------------ classification


def _sites(d: dict[str, Any]) -> list[str]:
    """Numbering-independent signature of the LS sites of a canonical graph."""
    sym = d["symbols"]
    out = [str(("A", s["kind"], sym[s["atoms"][0]])) for s in d["atom_stereo"]]
    for s in d["bond_stereo"]:
        a = s["atoms"]
        ends = tuple(sorted((sym[a[2]], sym[a[3]])))
        out.append(str(("B", s["kind"], ends, len(s.get("chain") or []))))
    return sorted(out)


def kind_of(row: dict[str, Any]) -> str:
    """The family of a species: radical, charge-separated, or closed-shell neutral."""
    if row.get("is_radical"):
        return "radical"
    if row.get("is_charged"):
        return "charge-separated"
    return "closed-shell"


def mismatch_class(row: dict[str, Any]) -> str:
    """Finer class of a from_smiles / from_geometry mismatch (the stereo ones)."""
    cat = row["geo_category"].split(" (")[0]
    if cat != "stereo" or "cj_smiles" not in row:
        return cat
    a, b = json.loads(row["cj_smiles"]), json.loads(row["cj_geo"])
    if any(s.get("chain") for s in a["bond_stereo"] + b["bond_stereo"]):
        return "stereo: chain LS"
    if _sites(a) != _sites(b):
        return "stereo: different LS sites (another Lewis structure)"
    return "stereo: same LS sites, different parity"


def partition_class(rows: list[dict[str, Any]]) -> str:
    """Category of an InChI/canonical disagreement, from the flags of its species."""
    if any(r.get("has_chain") for r in rows):
        return "chains"
    if any(r.get("has_pseudo") for r in rows):
        return "pseudoasymmetric centres"
    if any(r.get("has_amine_n") for r in rows):
        return "amine N"
    if any(r.get("is_radical") or r.get("is_charged") for r in rows):
        return "resonance or radical double bonds"
    if len({r.get("cj0_hash") for r in rows}) > 1:
        return "other: mobile-H tautomers (InChI merges different graphs)"
    return "other"


# ------------------------------------------------------------------- reproducers

HEAD = '''"""Reproducer: {title}

{detail}

Run in protomol's environment: `pixi run -e dev python {name}.py`. Exit status 1 means the
failure reproduces. Species {ident} of the stress test.
"""

import random
import sys
import warnings

import autogeom  # noqa: F401
import protomol as pm

warnings.simplefilter("ignore")
'''

RELABEL = """
def relabel_failures(gra, n=100):
    ref = pm.canonical_json(gra)
    rng = random.Random(0)
    bad = 0
    for _ in range(n):
        perm = list(range(len(gra.symbols)))
        rng.shuffle(perm)
        bad += pm.canonical_json(pm.relabel(gra, perm)) != ref
    return bad
"""


def _geo_lines(row: dict[str, Any]) -> str:
    return (
        f"SYMBOLS = {row['geo_symbols']!r}\n"
        f"COORDS = {row['geo_coords']!r}  # angstrom\n"
        "gs = pm.from_smiles(SMILES)\n"
        "geo = autogeom.Geometry(symbols=SYMBOLS, coordinates=COORDS, charge=gs.charge, spin=gs.spin)\n"
        "gg = pm.from_geometry(geo)\n"
    )


def repro_relabel(row: dict[str, Any], *, geo: bool) -> tuple[str, str]:
    """Body and expectation for a relabelling failure."""
    body = RELABEL + f"\nSMILES = {row['smiles']!r}\n"
    if geo:
        body += _geo_lines(row)
        body += "bad = relabel_failures(gg)\nprint('geometry graph: canonical_json changed under', bad, 'of 100 relabellings')\n"
    else:
        body += "gs = pm.from_smiles(SMILES)\nbad = relabel_failures(gs)\nprint('from_smiles graph: canonical_json changed under', bad, 'of 100 relabellings')\n"
    return (
        body + "sys.exit(1 if bad else 0)\n",
        "canonical_json is the same under every relabelling",
    )


def repro_mismatch(row: dict[str, Any]) -> tuple[str, str]:
    """Body for a from_smiles / from_geometry mismatch."""
    full = row.get("geo_compared") == "full"
    body = f"SMILES = {row['smiles']!r}\n" + _geo_lines(row)
    if full:
        body += "same = pm.canonical_json(gs) == pm.canonical_json(gg)\n"
        what = "equal canonical JSON"
    else:
        body += "same = pm.canonical_json(pm.without_stereo(gs)) == pm.canonical_json(pm.without_stereo(gg))\n"
        what = (
            "equal stereo-free canonical JSON (the SMILES leaves stereo undetermined)"
        )
    body += "print('from_smiles and from_geometry give', 'the same' if same else 'DIFFERENT', 'graphs')\nsys.exit(0 if same else 1)\n"
    return body, what


def repro_geo_error(row: dict[str, Any]) -> tuple[str, str]:
    """Body for a from_geometry error or timeout."""
    body = f"SMILES = {row['smiles']!r}\n" + _geo_lines(row)
    return body + "print('no error')\n", "from_geometry returns a graph"


def repro_partition(
    a: dict[str, Any], b: dict[str, Any], *, full: bool, same_inchi: bool
) -> tuple[str, str]:
    """Body for a disagreement between the canonical form and InChI."""
    wrap = "" if full else "pm.without_stereo"
    canon = (
        f"pm.canonical_json({wrap}(pm.from_smiles(s)))"
        if wrap
        else "pm.canonical_json(pm.from_smiles(s))"
    )
    body = (
        f"A, B = {a['smiles']!r}, {b['smiles']!r}\n"
        "from rdkit import Chem\n"
        "from rdkit.Chem import inchi\n\n"
        "def inchi_of(s):\n"
        f"    return inchi.MolToInchi(Chem.MolFromSmiles(s){'' if full else ', options="-SNon"'})\n\n"
        f"canon_equal = {canon.replace('(s)', '(A)')} == {canon.replace('(s)', '(B)')}\n"
        "inchi_equal = inchi_of(A) == inchi_of(B)\n"
        "print('InChI equal:', inchi_equal, '/ canonical form equal:', canon_equal)\n"
        "print(inchi_of(A))\nprint(inchi_of(B))\n"
        "sys.exit(1 if canon_equal != inchi_equal else 0)\n"
    )
    what = f"the canonical form and the InChI ({'standard' if full else 'without stereo'}) agree on whether the two species are the same"
    return body, what


def write_repro(
    directory: Path, name: str, title: str, detail: str, ident: str, body: str
) -> Path:
    """Write one reproducer script."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(
        HEAD.format(title=title, detail=detail, name=name, ident=ident) + "\n" + body
    )
    return path


def run_repro(path: Path) -> str:
    """Run a reproducer: 'reproduces', 'does not reproduce' or an error line."""
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if proc.returncode == 1:
        return "reproduces"
    if proc.returncode == 0:
        return "does NOT reproduce"
    return "script error: " + (proc.stderr.strip().splitlines() or ["?"])[-1][:120]


# --------------------------------------------------------------------------- report


def smallest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The species with the fewest atoms."""
    return min(rows, key=lambda r: r.get("n_atoms") or 10**9)


def table(header: list[str], body: list[list[Any]]) -> str:
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def provenance() -> list[str]:
    """Versions of what was tested."""
    from rdkit import rdBase
    from rdkit.Chem import inchi

    import protomol

    commit = "?"
    gitrepo = ROOT / "extern/stereomolgraph/.gitrepo"
    if gitrepo.exists():
        for line in gitrepo.read_text().splitlines():
            if line.strip().startswith("commit ="):
                commit = line.split("=")[1].strip()[:10]
    protomol_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return [
        f"protomol {protomol.__version__} (git {protomol_commit}); vendored stereomolgraph (`protomol` branch) at {commit}",
        f"RDKit {rdBase.rdkitVersion}, InChI {inchi.GetInchiVersion()}",
    ]


def build(rows: list[dict[str, Any]], results: Path, report: Path) -> str:
    """The generated part of the report (and the side files)."""
    out_dir = report.with_suffix("")
    repro_dir = out_dir / "repro"
    repros: list[tuple[str, str, str, str]] = []  # name, title, what's expected, status

    def add(
        name: str, title: str, detail: str, row: dict[str, Any], made: tuple[str, str]
    ) -> None:
        body, expected = made
        path = write_repro(
            repro_dir, name, title, f"{detail}\nExpected: {expected}.", row["id"], body
        )
        repros.append((name, title, f"`{row['smiles']}`", run_repro(path)))

    lines: list[str] = []
    n = len(rows)
    by_ds = Counter(r["dataset"] for r in rows)
    lines += [
        f"Results file: `{results}` ({n:,} species: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(by_ds.items()))
        + ").",
        "",
        *[f"- {p}" for p in provenance()],
        "",
    ]
    harness = [r for r in rows if "harness_error" in r]
    lines += [
        f"Harness errors (a bug in the test, not in protomol): {len(harness)}.",
        "",
    ]

    # -- front end errors
    lines += ["## From SMILES: errors"]
    errs = Counter(
        (r["dataset"], r["smiles_error"][:90]) for r in rows if "smiles_error" in r
    )
    errs += Counter(
        (r["dataset"], "canonical_json: " + r["canon_error"][:90])
        for r in rows
        if "canon_error" in r
    )
    lines += [
        table(
            ["dataset", "error", "species"],
            [[d, e, c] for (d, e), c in errs.most_common(15)],
        )
        if errs
        else "None.",
        "",
    ]
    for i, ((d, e), c) in enumerate(errs.most_common(5)):
        ex = smallest(
            [
                r
                for r in rows
                if r["dataset"] == d
                and (
                    r.get("smiles_error", "")[:90] == e
                    or "canonical_json: " + r.get("canon_error", "")[:90] == e
                )
            ]
        )
        body = f"SMILES = {ex['smiles']!r}\ngs = pm.from_smiles(SMILES)\nprint(pm.canonical_json(gs)[:80])\n"
        add(
            f"error-{i + 1}",
            f"error: {e}",
            f"{c} species in {d}.",
            ex,
            (body, "no error"),
        )

    # -- check 1
    lines += ["## Check 1: canonical JSON under random relabellings"]
    body = []
    for label, key in (("from_smiles graph", "smiles"), ("from_geometry graph", "geo")):
        checked = [r for r in rows if f"relabel_{key}_bad" in r]
        bad = [r for r in checked if r[f"relabel_{key}_bad"]]
        err = [r for r in rows if f"relabel_{key}_error" in r]
        chain = [r for r in bad if r.get("has_chain")]
        body.append(
            [
                label,
                f"{len(checked):,}",
                len(bad),
                len(chain),
                len(bad) - len(chain),
                len(err),
            ]
        )
    lines += [
        table(
            [
                "graph",
                "species checked",
                "species with a failure",
                "with a chain (allene/cumulene)",
                "without",
                "errors/timeouts",
            ],
            body,
        ),
        "",
    ]
    for key, geo in (("smiles", False), ("geo", True)):
        bad = [r for r in rows if r.get(f"relabel_{key}_bad")]
        for tag, members in (
            ("chain", [r for r in bad if r.get("has_chain")]),
            ("other", [r for r in bad if not r.get("has_chain")]),
        ):
            group = [r for r in members if not geo or "geo_coords" in r]
            if group:
                ex = smallest(group)
                what = (
                    "SMILES has a cumulene"
                    if tag == "chain"
                    else "no cumulene in the SMILES"
                )
                add(
                    f"relabel-{key}-{tag}",
                    f"canonical_json not invariant ({key} graph, {what})",
                    f"{len(group)} species.",
                    ex,
                    repro_relabel(ex, geo=geo),
                )
    lines += [""]

    # -- check 2
    lines += ["## Check 2: `from_smiles` and `from_geometry` of the same species"]
    geo_rows = [r for r in rows if "geo_category" in r]
    cats = Counter()
    for r in geo_rows:
        c = r["geo_category"]
        cats[
            (
                c
                if c.startswith(("no geometry", "match"))
                else mismatch_class(r)
                + (" (SMILES undetermined)" if "(SMILES undetermined)" in c else ""),
                r["dataset"],
            )
        ] += 1
    keys = sorted(
        {k for k, _ in cats},
        key=lambda k: (not k.startswith("match"), k.startswith("no geometry"), k),
    )
    lines += [
        table(
            ["category", "rgd1", "stjohn"],
            [[k, cats[(k, "rgd1")], cats[(k, "stjohn")]] for k in keys],
        ),
        "",
    ]
    lines += [
        'Full canonical JSON is compared when `is_determined(from_smiles(...))`, otherwise the stereo-free one ("SMILES undetermined").',
        "",
    ]
    mism = [
        r
        for r in geo_rows
        if (
            not r["geo_category"].startswith(("match", "no geometry"))
            and "geo_coords" in r
        )
        or r.get("geo_category") in ("no_lewis", "timeout", "error")
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in mism:
        c = r["geo_category"]
        key = (
            mismatch_class(r)
            if c.split(" (")[0] in ("stereo", "connectivity", "charge_spin")
            else c
        )
        groups[f"{key} / {kind_of(r)}"].append(r)
    lines += ["Mismatches by class and family (examples and reproducers below):", ""]
    ex_rows = []
    for i, (key, group) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ):
        ex = smallest(group)
        name = f"geo-{i + 1:02d}"
        if ex["geo_category"] in ("no_lewis", "timeout", "error"):
            add(
                name,
                f"from_geometry: {key}",
                f"{len(group)} species; {ex.get('geo_error', '')[:150]}",
                ex,
                repro_geo_error(ex),
            )
        else:
            add(
                name,
                f"from_smiles vs from_geometry: {key}",
                f"{len(group)} species.",
                ex,
                repro_mismatch(ex),
            )
        ex_rows.append([key, len(group), f"`{ex['smiles']}`", f"`{name}`"])
    lines += [
        table(["class / family", "species", "smallest example", "reproducer"], ex_rows),
        "",
    ]

    # -- check 3
    lines += ["## Check 3: equal standard InChI, equal canonical JSON"]
    by_inchi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "inchi" in r and "cj_hash" in r:
            by_inchi[r["inchi"]].append(r)
    dup = {k: v for k, v in by_inchi.items() if len(v) > 1}
    where = Counter()
    bad_dup = {}
    for k, v in dup.items():
        ds = {r["dataset"] for r in v}
        scope = "across the datasets" if len(ds) > 1 else f"within {next(iter(ds))}"
        where[(scope, len({r["cj_hash"] for r in v}) > 1)] += 1
        if len({r["cj_hash"] for r in v}) > 1:
            bad_dup[k] = v
    lines += [
        table(
            ["scope", "InChI shared by >1 species", "with different canonical JSON"],
            [
                [s, where[(s, False)] + where[(s, True)], where[(s, True)]]
                for s in sorted({s for s, _ in where})
            ],
        )
        if where
        else "No shared InChI.",
        "",
    ]
    for i, (k, v) in enumerate(
        sorted(bad_dup.items(), key=lambda kv: min(r["n_atoms"] for r in kv[1]))[:5]
    ):
        a = v[0]
        b = next(r for r in v if r["cj_hash"] != a["cj_hash"])
        add(
            f"duplicate-{i + 1}",
            "equal InChI, different canonical JSON",
            f"`{k}`",
            a,
            repro_partition(a, b, full=True, same_inchi=True),
        )
    lines += [""]

    # -- check 4
    lines += ["## Check 4: partitions, canonical form vs InChI"]
    part_rows = []
    for label, ikey, ckey, full in (
        ("stereo-free", "inchi_nostereo", "cj0_hash", False),
        ("full", "inchi", "cj_hash", True),
    ):
        i2c: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        c2i: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in rows:
            if ikey in r and ckey in r:
                i2c[r[ikey]][r[ckey]].append(r)
                c2i[r[ckey]][r[ikey]].append(r)
        for direction, mapping in (
            ("same InChI, different canonical form", i2c),
            ("same canonical form, different InChI", c2i),
        ):
            split = {k: v for k, v in mapping.items() if len(v) > 1}
            cat_groups: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
            for v in split.values():
                members = [r for sub in v.values() for r in sub]
                cat_groups[partition_class(members)].append(members)
            for cat, gl in sorted(cat_groups.items()):
                ex_members = min(gl, key=lambda m: min(r["n_atoms"] for r in m))
                a = min(ex_members, key=lambda r: r["n_atoms"])
                other = ikey if direction.startswith("same canonical") else ckey
                b = next((r for r in ex_members if r.get(other) != a.get(other)), None)
                name = f"partition-{label.replace('-', '')}-{'canon' if direction.startswith('same canonical') else 'inchi'}-{cat.split()[0].strip(':')}"
                if b is not None and not any(rp[0] == name for rp in repros):
                    add(
                        name,
                        f"{label}: {direction} ({cat})",
                        f"{len(gl)} classes.",
                        a,
                        repro_partition(
                            a,
                            b,
                            full=full,
                            same_inchi=direction.startswith("same InChI"),
                        ),
                    )
                part_rows.append(
                    [
                        label,
                        direction,
                        cat,
                        len(gl),
                        f"`{a['smiles']}`" + (f" / `{b['smiles']}`" if b else ""),
                    ]
                )
        lines_total = (
            sum(len(v) > 1 for v in i2c.values()),
            sum(len(v) > 1 for v in c2i.values()),
        )
        lines += [
            f"- {label}: {len(i2c):,} InChI classes and {len(c2i):,} canonical classes; {lines_total[0]} InChI classes are split by the canonical form, {lines_total[1]} canonical classes are split by InChI."
        ]
    lines += [
        "",
        table(
            ["partition", "disagreement", "category", "classes", "example"], part_rows
        )
        if part_rows
        else "No disagreements.",
        "",
    ]

    # -- check 5
    lines += ["## Check 5: timing against heavy-atom count"]
    bins: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if "t_canon" in r and r.get("n_heavy"):
            bins[r["n_heavy"]].append(r["t_canon"] + r.get("t_from_smiles", 0))
    trows = [
        [
            h,
            len(v),
            f"{statistics.median(v) * 1000:.0f}",
            f"{sorted(v)[int(0.95 * (len(v) - 1))] * 1000:.0f}",
            f"{max(v) * 1000:.0f}",
        ]
        for h, v in sorted(bins.items())
    ]
    lines += [
        "`from_smiles` + `canonical_json` + `canonical_json(without_stereo)`, in ms:",
        "",
        table(["heavy atoms", "species", "median", "95 %", "max"], trows),
        "",
    ]
    timed = sorted(
        (r for r in rows if "t_canon" in r),
        key=lambda r: -(r["t_canon"] + r.get("t_from_smiles", 0)),
    )[:100]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "slowest-100.csv").open("w") as handle:
        handle.write(
            "id,n_heavy,n_atoms,t_from_smiles,t_canon,t_from_geometry,t_canon_geo,smiles\n"
        )
        for r in timed:
            handle.write(
                ",".join(
                    str(r.get(k, ""))
                    for k in (
                        "id",
                        "n_heavy",
                        "n_atoms",
                        "t_from_smiles",
                        "t_canon",
                        "t_from_geometry",
                        "t_canon_geo",
                    )
                )
                + f',"{r["smiles"]}"\n'
            )
    lines += [
        f"The slowest 100 are in `{out_dir.name}/slowest-100.csv`; the slowest is {timed[0]['id']} ({timed[0]['t_canon']:.2f} s for the canonical form)."
        if timed
        else "",
        "",
    ]
    timeouts = sum(
        1 for r in rows for k in r if k.endswith("error") and r[k] == "timeout"
    )
    lines += [f"Phases that hit the time cap: {timeouts}.", ""]

    # -- reproducers
    lines += [
        "## Reproducers",
        f"In `{out_dir.name}/repro/` (run each with `pixi run -e dev python <file>` in protomol; exit status 1 = reproduces):",
        "",
    ]
    lines += [
        table(
            ["script", "failure", "species", "status when this report was made"],
            [[f"`{a}.py`", b, c, d] for a, b, c, d in repros],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path.home() / "proj/pm/reports/2026-11-canonical-stress-test.md",
    )
    args = parser.parse_args()
    rows = load(args.results.expanduser())
    kept = ""
    if args.report.exists():
        text = args.report.read_text()
        if MARKER in text:
            kept = text.split(MARKER, 1)[1]
    head = "# Canonical-form stress test on RGD1 and stjohn-radicals\n\nGenerated by `protomol/benchmarks/canonical_report.py`; do not edit above the marker line at the end.\n\n"
    generated = build(rows, args.results, args.report.expanduser())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        head + generated + "\n" + MARKER + (kept or "\n\n## For Maxim\n\n(to write)\n")
    )
    print(f"report -> {args.report}")


if __name__ == "__main__":
    sys.exit(main())
