"""The vendored stereomolgraph imports from its own path, not a top-level package."""

import sys


def test_vendored_stereomolgraph_imports() -> None:
    """The vendored copy imports without a top-level `stereomolgraph`."""
    from protomol._vendor.stereomolgraph import StereoMolGraph  # noqa: PLC0415

    assert StereoMolGraph.__module__.startswith("protomol._vendor.stereomolgraph")
    assert "stereomolgraph" not in sys.modules
