from importlib.metadata import version

import deskgraph


def test_version_comes_from_package_metadata() -> None:
    # Lineage will report this as the engine version, so it must match pyproject.toml.
    assert deskgraph.__version__ == version("deskgraph")
