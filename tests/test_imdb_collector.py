import pytest
from pathlib import Path
import os
import sys

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the DataCollector
from src.data.imdb_collector import DataCollector

def test_basic_init():
    """Test basic initialization"""
    collector = DataCollector()
    assert isinstance(collector, DataCollector)