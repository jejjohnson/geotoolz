import xrtoolz


def test_import():
    assert xrtoolz is not None


def test_version():
    assert isinstance(xrtoolz.__version__, str)
