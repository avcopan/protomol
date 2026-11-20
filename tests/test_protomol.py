"""protomol tests."""

import protomol


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(protomol.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert protomol.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert protomol.greet_jim() == "Hello, Jim!"
