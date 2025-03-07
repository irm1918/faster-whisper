import os

import pytest


@pytest.fixture
def data_dir():
    """
    This fixture function returns the absolute path to the data directory.
    
    Returns:
        str: The absolute path to the data directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def jfk_path(data_dir):
    """
    This fixture function returns the absolute path to the 'jfk.flac' file in the data directory.
    
    Args:
        data_dir (str): The absolute path to the data directory.
    
    Returns:
        str: The absolute path to the 'jfk.flac' file.
    """
    return os.path.join(data_dir, "jfk.flac")
