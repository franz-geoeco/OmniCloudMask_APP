# tests/test_file_utils.py
import os
import pytest
from file_utils import find_raster_files, is_multiband_file, group_files_by_date_tile

def test_find_raster_files(tmp_path):
    # Create temporary test files
    test_files = [
        tmp_path / "test1.tif",
        tmp_path / "test2.tiff",
        tmp_path / "test3.TIF",
        tmp_path / "test4.txt"  # Should not be found
    ]
    for file in test_files:
        file.touch()
    
    # Test function
    found_files = find_raster_files(tmp_path)
    
    # Verify results
    assert len(found_files) == 3
    assert str(tmp_path / "test1.tif") in found_files
    assert str(tmp_path / "test2.tiff") in found_files
    assert str(tmp_path / "test3.TIF") in found_files
    assert str(tmp_path / "test4.txt") not in found_files

def test_is_multiband_file(monkeypatch):
    # Mock rasterio.open
    class MockDataset:
        def __init__(self, count):
            self.count = count
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    # Test with multiband file
    monkeypatch.setattr("rasterio.open", lambda x: MockDataset(3))
    assert is_multiband_file("fake_path.tif") is True
    
    # Test with single band file
    monkeypatch.setattr("rasterio.open", lambda x: MockDataset(1))
    assert is_multiband_file("fake_path.tif") is False
    
    # Test with exception handling
    def raise_exception(*args):
        raise Exception("Test exception")
    
    monkeypatch.setattr("rasterio.open", raise_exception)
    assert is_multiband_file("fake_path.tif") is False