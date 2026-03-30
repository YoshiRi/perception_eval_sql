"""Pytest configuration for evaluation_dashboard_app tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that require a live service or network (opt-in)",
    )
