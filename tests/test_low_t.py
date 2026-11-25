"""The low-T scheme's sites: which sites of a species get an LS (P16, Phase 3).

The rules come from the user's chapter, "Accounting for Stereochemistry" (ch12 L1491-
1700): tetrahedral atoms and planar bonds are rigid features (L1537-1567); amines and
resonating planar bonds are semi-rigid features, which the low-T scheme keeps (L1655);
a pyramidal atom is a tetrahedral feature with a lone pair, and whether it has one
depends on the resonance structure (L1562-1567); alkenyl radical bonds are left out and
their stereoisomers lumped (L1648-1651).

Each case gives a SMILES, the sites the low-T scheme puts an LS on, and what Phase 1's
approximation gave (`today`, before the low-T rules; kept as the record of what
changed). A site is written `C1` (a `Tetrahedral`), `N1lp` (one with a lone pair) or
`C1=C2` (a bond LS), followed by `?` if it is undetermined or `:parity` if it is fixed
(given in the SMILES, or trivial). `Tetrahedral` sites on an atom with two or more H
(CH3, CH2, NH2) are left out of the strings: they are always there and trivial.
"""

import re
from dataclasses import dataclass

import pytest

import protomol as pm
from protomol._types import neighbors


@dataclass(frozen=True)
class Case:
    """A SMILES, its low-T sites, and what Phase 1's approximation gave."""

    smiles: str
    low_t: str
    today: str
    why: str


CASES = [
    # Stereocentres, rings included
    Case("C[C@H](O)CC", "C1:-1", "C1:-1", "a given stereocentre"),
    Case("CC(O)CC", "C1?", "C1?", "an unspecified stereocentre"),
    Case("CC(C)C", "C1:1", "C1:1", "two equal methyls: trivial"),
    Case("C1CC(C)CC(C)C1", "C2? C5?", "C2? C5?", "1,3-dimethylcyclohexane"),
    Case(
        "CC1CCC(C)CC1",
        "C1? C4?",
        "C1:1 C4:1",
        "1,4-dimethylcyclohexane: cis/trans, though each centre alone looks trivial",
    ),
    Case("C[C@H]1CC[C@@H](C)CC1", "C1:1 C4:1", "C1:1 C4:1", "given cis or trans"),
    Case(
        "OC(=O)C(O)C(O)C(O)C(=O)O",
        "C3? C5? C7?",
        "C3? C5:1 C7?",
        "open question 8: the pseudoasymmetric C5 is stereogenic in the meso forms",
    ),
    Case(
        "OC(=O)[C@@H](O)C(O)[C@H](O)C(=O)O",
        "C3:1 C5:1 C7:-1",
        "C3:1 C5:1 C7:-1",
        "the chiral diastereomer: C5 is trivial there",
    ),
    Case(
        "OC(=O)[C@@H](O)C(O)[C@@H](O)C(=O)O",
        "C3:1 C5? C7:1",
        "C3:1 C5? C7:1",
        "a meso form: C5 is r or s",
    ),
    Case("C[N+](CC)(F)Cl", "N1?", "N1?", "an ammonium centre (no lone pair)"),
    # Pyramidal atoms
    Case("CN(CC)CCC", "N1lp?", "N1lp?", "a tertiary amine"),
    Case("CN(C)C", "N1lp:1", "N1lp:1", "trimethylamine: trivial"),
    Case("CN(F)Cl", "N1lp?", "N1lp?", "an amine with three different substituents"),
    Case(
        "C[N@](F)Cl",
        "N1lp:1",
        "N1lp?",
        "open question 9: RDKit's parser drops a chiral tag on N",
    ),
    Case("C[N@@](F)Cl", "N1lp:-1", "N1lp?", "open question 9, the other parity"),
    Case(
        "[N@](C)(F)Cl",
        "N0lp:1",
        "N0lp?",
        "atom-first: the same parity as C[N@](F)Cl, as RDKit reads [P@](C)(F)Cl",
    ),
    Case("CC(=O)[N@](C)CC", "", "", "a tag on an amide N is dropped"),
    Case("CC1CN1C", "C1? N3lp?", "C1? N3lp?", "an aziridine N"),
    Case("CN(O)CC", "N1lp?", "N1lp?", "a hydroxylamine N"),
    Case("CN(N)CC", "N1lp?", "N1lp?", "a hydrazine N"),
    Case("CC(=O)N(C)CC", "", "", "an amide N is conjugated: planar"),
    Case("CN(CC)C=C", "C4=C5:0", "C4=C5:0", "an enamine N is conjugated"),
    Case("CN(CC)N=O", "", "", "a nitrosamine N is conjugated"),
    Case(
        "CN(CC)[O]",
        "",
        "N1lp?",
        "a nitroxide N: its lone pair is conjugated with the radical",
    ),
    Case(
        "[CH2]N(C)CC",
        "",
        "N1lp?",
        "an alpha-amino radical's N: its lone pair is conjugated with the radical",
    ),
    Case("CP(CC)CCC", "P1lp?", "", "a phosphine (inversion barrier ≈ 35 kcal/mol)"),
    Case("CP(F)Cl", "P1lp?", "", "a phosphine"),
    Case("C[P@](F)Cl", "P1lp:1", "P1lp:1", "a given phosphine"),
    Case("CS(=O)CC", "S1lp?", "", "a sulfoxide, S=O form"),
    Case("C[S+]([O-])CC", "S1lp?", "", "a sulfoxide, S+-O- form"),
    Case("C[S@](=O)CC", "S1lp:1", "S1lp:1", "a given sulfoxide"),
    Case("C[S+](CC)CCC", "S1lp?", "", "a sulfonium ion"),
    Case("C[O+](CC)CCC", "", "", "an oxonium ion inverts too fast"),
    Case("C[CH-]CC", "", "", "a carbanion inverts too fast"),
    # Double bonds
    Case("CC=CC", "C1=C2?", "C1=C2?", "an unspecified double bond"),
    Case("C/C=C/C", "C1=C2:0", "C1=C2:0", "a given double bond"),
    Case("CC=C(C)C", "C1=C2:0", "C1=C2:0", "a trivial double bond"),
    Case("CC=NC", "C1=N2?", "C1=N2?", "an imine (N inversion ≈ 25 kcal/mol)"),
    Case("CC=NO", "C1=N2?", "C1=N2?", "an oxime"),
    Case("CC=C=CC", "C1=C3?", "C1=C3?", "an allene's chain LS stays (P45)"),
    Case("C=CC=C", "C0=C1:0 C2=C3:0", "C0=C1:0 C2=C3:0", "butadiene: C2-C3 rotates"),
    # Resonating double bonds in radicals
    Case("C=C[CH2]", "C0=C1:0 C1=C2:0", "C0=C1:0 C1=C2:0", "allyl"),
    Case("[CH2]C=CC", "C0=C1:0 C1=C2?", "C0=C1:0 C1=C2?", "but-2-en-1-yl"),
    Case("C[CH]C=CC", "C1=C2? C2=C3?", "C1=C2? C2=C3?", "pent-3-en-2-yl"),
    Case("CC=C[O]", "C1=C2?", "C1=C2?", "a vinoxy radical"),
    Case(
        "C[C]=CC",
        "",
        "C1=C2?",
        "an alkenyl radical: its bond is left out and its isomers lumped (L1648)",
    ),
    Case("[CH]=CC", "", "C0=C1?", "an alkenyl radical"),
    Case("C1=[C]CCCC1", "C0=C1:0", "C0=C1:0", "an alkenyl radical in a small ring"),
    Case("C[C]=C=CC", "", "C1=C3?", "an allenyl radical: its chain LS is left out"),
    Case("C=C[C]=CC", "C0=C1:0", "C0=C1:0 C2=C3?", "a dienyl radical's alkenyl bond"),
    # Charged resonance
    Case("CC(=O)[O-]", "", "", "acetate: neither C-O bond has a substituent on O"),
    Case("C[N+](=O)[O-]", "", "", "nitromethane"),
    Case("CC=[O+][O-]", "C1=O2?", "C1=O2?", "a Criegee intermediate: syn or anti"),
    Case(
        "C=CC=[O+][O-]",
        "C0=C1:0 C2=O3?",
        "C0=C1:0 C2=O3?",
        "acrolein oxide: the C=O+ bond (C1-C2, s-cis/s-trans, rotates)",
    ),
    Case("CC=[N+]([O-])C", "C1=N2?", "C1=N2?", "a nitrone"),
    # Resonating double bonds in charged radicals
    Case(
        "CC=C[CH][N+](=O)[O-]",
        "C1=C2? C2=C3? C3=N4:0",
        "C1=C2?",
        "an allylic radical with a nitro group (stereomolgraph's radical resonance)",
    ),
    Case(
        "C[CH]C=CC=[O+][O-]",
        "C1=C2? C2=C3? C3=C4? C4=O5?",
        "C2=C3? C4=O5?",
        "an allylic radical with a carbonyl oxide",
    ),
]

# The cases each later rule fixes, until it lands
_TRIVIAL = "trivial sites are tested against the other sites' configurations"
PENDING: dict[str, str] = {
    "CC1CCC(C)CC1": _TRIVIAL,
    "OC(=O)C(O)C(O)C(O)C(=O)O": _TRIVIAL,
}
# The round trip fails where `from_geometry`'s Lewis structure is another species than
# the SMILES's: the geometry of this carbonyl oxide reads as a dienyl peroxy radical
# (C=CC=CO[O]), which has no C2=C3 or C4=O5 bond
PENDING_GEOMETRY = {
    "C[CH]C=CC=[O+][O-]": "from_geometry reads it as a dienyl peroxy radical",
}


def _sites(g: pm.StereoMolGraph) -> str:
    """The LS of `g` as a case's string (see the module docstring)."""
    sym = g.symbols
    tokens = []
    for ls in g.atom_stereo:
        i = ls.atoms[0]
        if sum(sym[j] == "H" for j in neighbors(g, i)) >= 2:
            continue
        lp = "lp" if None in ls.atoms else ""
        tokens.append(((i,), f"{sym[i]}{i}{lp}", ls.parity))
    for ls in g.bond_stereo:
        i, j = sorted(ls.atoms[2:4])
        tokens.append(((i, j), f"{sym[i]}{i}={sym[j]}{j}", ls.parity))
    return " ".join(
        f"{label}?" if parity is None else f"{label}:{parity}"
        for _, label, parity in sorted(tokens)
    )


def _labels(g: pm.StereoMolGraph) -> list[str]:
    """The sites of `g` without their parities."""
    return re.sub(r"\?|:-?\d", "", _sites(g)).split()


def _params(cases: list[Case], pending: dict[str, str]) -> list:
    return [
        pytest.param(
            case,
            id=case.smiles,
            marks=[pytest.mark.xfail(reason=pending[case.smiles], strict=True)]
            if case.smiles in pending
            else [],
        )
        for case in cases
    ]


@pytest.mark.parametrize("case", _params(CASES, PENDING))
def test_from_smiles_sites(case: Case) -> None:
    """`from_smiles` puts an LS on exactly the low-T sites."""
    assert _sites(pm.from_smiles(case.smiles)) == case.low_t, case.why


@pytest.mark.parametrize("case", _params(CASES, PENDING_GEOMETRY))
def test_from_geometry_sites(case: Case) -> None:
    """`from_geometry(embed(g))` has the sites of `g`, and its determined parities."""
    gra = pm.from_smiles(case.smiles)
    back = pm.from_geometry(pm.embed(gra))
    found = (
        pm.stereo_at(back, pm.site(ls))
        for ls in (*gra.atom_stereo, *gra.bond_stereo)
        if ls.parity is None
    )
    filled = pm.with_stereo(gra, *(ls for ls in found if ls is not None))
    assert _labels(back) == _labels(gra), case.why
    assert back == filled, case.why
