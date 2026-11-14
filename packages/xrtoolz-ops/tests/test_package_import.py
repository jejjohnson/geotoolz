import xrops


def test_import():
    assert xrops is not None


def test_version():
    assert isinstance(xrops.__version__, str)
