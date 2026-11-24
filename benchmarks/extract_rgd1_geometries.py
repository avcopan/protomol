"""Extract the RGD1 reactant/product geometries for the canonical stress test.

Stage 1 of `canonical_stress.py`. It needs qccodec and the Gaussian plugin, which are not
in protomol's environment, so run it with the Python of the study-hifithermkin test env:

    ~/code/am/study-hifithermkin/.pixi/envs/test/bin/python \
        benchmarks/extract_rgd1_geometries.py --sample      # the 1 % sample
    ... benchmarks/extract_rgd1_geometries.py               # every species

Species are the `molN` of `RandP_smiles.txt` (inside `RGD1_rawoutput.zip`, read in place:
nothing is unpacked). The geometry of `molN` is the last frame of `raw_output/RP/molN.out`,
read with the Gaussian plugin. A species is skipped (status other than "ok", no
coordinates) when it has no log in the sweep CSV, its log failed to parse, or its log has
NImag > 0 (the `Gaussian parse sweep` in `~/proj/am/DATA.md`).

Output: JSON lines (gzipped), one per species: `{id, status, symbols, coords}` with `id`
``molN`` and coordinates in angstrom. Resumable: a second run skips the ids in the file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "code/am/data"))
RGD1_ZIP = DATA / "rgd1/RGD1_rawoutput.zip"
SWEEP = DATA / "sweeps/gaussian-plugin-2026-11-24.csv"
BOHR = 0.529177210903  # angstrom per bohr (CODATA 2018)
_zip: zipfile.ZipFile | None = None


def in_sample(species_id: str) -> bool:
    """The 1 % sample: a stable hash of the id (the same rule as canonical_stress.py)."""
    return hashlib.sha1(species_id.encode()).digest()[0] % 100 == 0


def read_geometry(mol: str) -> dict[str, object]:
    """Decode one RP log in a worker."""
    global _zip  # noqa: PLW0603
    from qccodec import get_codec
    from qcdata import CalcType

    if _zip is None:
        _zip = zipfile.ZipFile(RGD1_ZIP)
    text = _zip.read(f"raw_output/RP/{mol}.out").decode(errors="replace")
    try:
        data = get_codec("gaussian").parse(CalcType.optimization, stdout=text)
        struct = data.frames[-1].structure
        coords = [[round(float(x) * BOHR, 6) for x in row] for row in struct.geometry]
        return {
            "id": mol,
            "status": "ok",
            "symbols": [str(s) for s in struct.symbols],
            "coords": coords,
        }
    except Exception as exc:  # noqa: BLE001
        return {"id": mol, "status": f"error: {type(exc).__name__}"}


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", action="store_true", help="only the 1 % sample")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (
        args.out
        or DATA
        / f"sweeps/rgd1-rp-geometries{'-sample' if args.sample else ''}.jsonl.gz"
    )

    with zipfile.ZipFile(RGD1_ZIP) as archive:
        lines = archive.read("raw_output/RandP_smiles.txt").decode().splitlines()[1:]
    species = [line.split()[1] for line in lines if line.strip()]
    if args.sample:
        species = [s for s in species if in_sample(f"rgd1:{s}")]

    sweep: dict[str, tuple[str, str]] = {}
    with SWEEP.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "/RP/" in row["id"]:
                mol = row["id"].rsplit("/", 1)[1].removesuffix(".out")
                sweep[mol] = (row["status"], row["nimag"])

    done: set[str] = set()
    if out.exists():
        with gzip.open(out, "rt") as handle:
            done = {json.loads(line)["id"] for line in handle}

    todo, skipped = [], []
    for mol in species:
        if mol in done:
            continue
        status, nimag = sweep.get(mol, ("missing", ""))
        if status == "ok" and nimag == "0":
            todo.append(mol)
        elif status == "ok":
            skipped.append({"id": mol, "status": f"skipped: NImag={nimag}"})
        else:
            skipped.append({"id": mol, "status": f"skipped: {status}"})

    # Multi-member gzip: appending a new member is valid and keeps a resume cheap.
    with gzip.open(out, "at") as handle, ProcessPoolExecutor(args.workers) as pool:
        for row in skipped:
            handle.write(json.dumps(row) + "\n")
        for row in pool.map(read_geometry, todo, chunksize=64):
            handle.write(json.dumps(row) + "\n")
    print(
        f"{len(species)} species, {len(done)} already done, {len(skipped)} skipped, "
        f"{len(todo)} read -> {out}"
    )


if __name__ == "__main__":
    sys.exit(main())
