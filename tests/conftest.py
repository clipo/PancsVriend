"""pytest configuration: the `slow` marker.

Tests marked @pytest.mark.slow run whole simulations (minutes) and are
SKIPPED by default so the suite stays fast after an edit. Run them with

    python -m pytest tests/ --runslow            # everything
    python -m pytest tests/ --runslow -m slow    # only the slow ones
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked slow (whole-simulation validations)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: whole-simulation validation, minutes; needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow validation; pass --runslow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
