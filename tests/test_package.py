import importlib


def test_core_package_is_importable() -> None:
    assert importlib.import_module("core").__package__ == "core"
