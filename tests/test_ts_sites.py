"""Tests for TS sites (P67-P74): the reacting core, the `ts` slots, the test reactions.

The reactions and their expected TS graphs are the tables of the TS-sites guide
(protomol's planning repo, `guides/2026-11-25-ts-sites.md`); atom indices are those of
`from_smiles` on the reactant side.
"""

import pytest

import protomol as pm
from protomol._ts import _reacting_core, _ts_sites


def _channels(rct: str, prd: str) -> list[pm.StereoReactionGraph]:
    return pm.ts_graphs(pm.from_smiles(rct), pm.from_smiles(prd))


@pytest.mark.parametrize(
    ("rct", "prd", "core", "sites"),
    [
        # 1: SN2, C0 attacked by I2 with Br1 leaving
        ("CBr.[I-]", "CI.[Br-]", {0, 1, 2}, (0,)),
        # 3: rotation about C1=C2, no bond change: the bond's ends. The allyl
        # C0-C1 bond's trivial PlanarBond (unchanged) has an end in the core too
        (r"[CH2]/C=C\C", r"[CH2]/C=C/C", {1, 2}, ((0, 1), (1, 2))),
        # 5: H8 abstracted from C1 by O4
        ("CCCC.[OH]", "C[CH]CC.O", {1, 4, 8}, (1,)),
        # 8: OH adds to C1 of the C1=C2 stereo bond: the centre and the bond
        ("C/C=C/C.[OH]", "CC(O)[CH]C", {1, 4}, (1, (1, 2))),
    ],
)
def test_reacting_core(rct: str, prd: str, core: set[int], sites: tuple) -> None:
    """The core is the atoms of the changing bonds, else the changing site's atoms."""
    for srg in _channels(rct, prd):
        assert _reacting_core(srg) == core
        plain = core if srg.breaking or srg.forming else set()
        assert _reacting_core(pm.without_stereo(srg)) == plain
        assert _ts_sites(srg) == sites
