"""Register the `slow` marker (`verify` covers the same ground as the slow
tests, but pytest should not warn about a marker it does not know)."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: full golden suite over all 38 splits (~40 min)")
