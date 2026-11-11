def pytest_configure(config):
    config.addinivalue_line("markers", "slow: learning checks that train for a few minutes on a GPU")
