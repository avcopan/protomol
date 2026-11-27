"""`stereoisomers` for species (P65, Step N8)."""

import pytest

import protomol as pm

# StereoMolGraph SI-001 (Papusha & Leonhard 2026): bicyclopropyl-2,2',3,3'-tetra-
# carboxylic acid, with 10 stereoisomers. Its two ring-joining C are stereocentres in
# some configurations of the other four and not in others.
SI_001 = "OC(=O)C1C(C(=O)O)C1C1C(C(=O)O)C1C(=O)O"


def test_si_001() -> None:
    """Verify: the 10 stereoisomers of StereoMolGraph SI-001, all distinct."""
    isos = pm.stereoisomers(pm.from_smiles(SI_001))
    assert len(isos) == 10
    assert all(pm.is_determined(smg) for smg in isos)
    assert len({pm.canonical_json(smg) for smg in isos}) == 10
    assert all(pm.low_t(smg) == smg for smg in isos)


@pytest.mark.parametrize(
    ("smiles", "count"),
    [
        ("CC(O)CC", 2),  # one centre
        ("CC(O)C(O)C", 3),  # two centres, one meso form
        ("CC=CC", 2),  # (E) and (Z)
        ("CC(O)C=CC", 4),  # a centre and a planar bond
        ("CCO", 1),  # no sites
        ("C[C@H](O)CC", 1),  # determined
        ("C[C@H](O)C(O)C", 2),  # one determined, one open
        pytest.param(
            "CC=C=CC",
            2,
            marks=[
                pytest.mark.xfail(
                    reason="the back-end drops chain LS, so deduplicate merges the "
                    "allene's two (P45)",
                    strict=True,
                ),
                pytest.mark.filterwarnings(
                    "ignore:stereomolgraph can't hold a chain LS"
                ),
            ],
        ),
    ],
)
def test_count(smiles: str, count: int) -> None:
    """The number of distinct stereoisomers, both mirror images kept."""
    isos = pm.stereoisomers(pm.from_smiles(smiles))
    assert len(isos) == count
    assert all(pm.is_determined(smg) for smg in isos)


def test_determined_gives_itself() -> None:
    """A determined graph gives itself."""
    gra = pm.from_smiles("C[C@@H](O)/C=C/C")
    assert pm.stereoisomers(gra) == [gra]


def test_determined_ls_kept() -> None:
    """A determined LS keeps its parity in every stereoisomer."""
    gra = pm.from_smiles("C[C@H](O)C(O)C")
    fixed = pm.stereo_at(gra, 1)
    assert all(pm.stereo_at(smg, 1) == fixed for smg in pm.stereoisomers(gra))


def test_mol_graph_gets_low_t_sites() -> None:
    """A `MolGraph` is expanded at its low-T sites, with its atom indices."""
    smg = pm.from_smiles("CC(O)CC")
    isos = pm.stereoisomers(pm.without_stereo(smg))
    assert isos == pm.stereoisomers(smg)
    assert [iso.atom_stereo[1].parity for iso in isos] == [1, -1]


def test_mirror_images() -> None:
    """A chiral species' stereoisomers come with their mirror images."""
    isos = pm.stereoisomers(pm.from_smiles("CC(O)C(O)C"))
    jsons = {pm.canonical_json(smg) for smg in isos}
    assert all(pm.canonical_json(pm.enantiomer(smg)) in jsons for smg in isos)
    assert sum(pm.is_chiral(smg) for smg in isos) == 2


def test_not_a_graph_rejected() -> None:
    """Reaction graphs are tested with TS sites (tests/test_ts_sites.py)."""
    with pytest.raises(TypeError, match="takes a graph"):
        pm.stereoisomers("CC")  # ty: ignore[no-matching-overload]
