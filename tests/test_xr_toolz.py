import xr_toolz


def test_import():
    assert xr_toolz is not None


def test_version():
    assert isinstance(xr_toolz.__version__, str)
