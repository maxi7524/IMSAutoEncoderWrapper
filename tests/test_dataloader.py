import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
import m2aia as m2
from ims_contrastive_model.dataloader import IMSPyTorchDataset

@pytest.fixture
def mock_m2aia_reader():
    """
    Creates a mock object that mimics m2aia.ImzMLReader behavior
    without requiring physical .imzML files.
    """
    reader = MagicMock(spec=m2.ImzMLReader)
    
    # Mocking GetMetaData to return expected dictionary keys
    reader.GetMetaData.return_value = {
        'm2aia.xs.min': '100.0',
        'm2aia.xs.max': '200.0',
        'pixel size z': '1.0'
    }
    
    # Mocking GetNumberOfSpectra
    reader.GetNumberOfSpectra.return_value = 10
    
    # Mocking GetSpectrum(idx) -> (mz_array, intensity_array)
    # Returning random data to simulate a mass spectrum
    def mock_get_spectrum(idx):
        if idx >= 10:
            raise IndexError("Out of bounds")
        mzs = np.linspace(100.0, 200.0, 50)
        intensities = np.random.rand(50).astype(np.float32)
        return mzs, intensities

    reader.GetSpectrum.side_effect = mock_get_spectrum
    return reader

def test_dataset_initialization_with_defaults(mock_m2aia_reader):
    """Test if dataset initializes correctly using metadata from m2aia object."""
    dataset = IMSPyTorchDataset(m2aia_img=mock_m2aia_reader)
    
    # Check if properties match the mocked metadata
    assert dataset.mz_min == 100.0
    assert dataset.mz_max == 200.0
    assert dataset.mz_resolution == 1.0
    assert len(dataset.grid) > 0

def test_dataset_custom_grid(mock_m2aia_reader):
    """Test if dataset respects manually provided mz range and resolution."""
    custom_res = 0.5
    dataset = IMSPyTorchDataset(
        m2aia_img=mock_m2aia_reader,
        mz_min=150.0,
        mz_max=160.0,
        mz_resolution=custom_res
    )
    
    assert dataset.mz_min == 150.0
    assert dataset.mz_max == 160.0
    # Grid should be roughly (160-150)/0.5 + 1 elements
    expected_len = int((160.0 - 150.0) / custom_res) + 1
    assert len(dataset.grid) == expected_len

def test_getitem_tensor_output(mock_m2aia_reader):
    """Test if __getitem__ returns a valid PyTorch tensor with correct shape."""
    dataset = IMSPyTorchDataset(mock_m2aia_reader, mz_resolution=1.0)
    sample = dataset[0]
    
    assert isinstance(sample, torch.Tensor)
    assert sample.dtype == torch.float32
    assert sample.shape[0] == len(dataset.grid)

def test_resampling_integrity(mock_m2aia_reader):
    """Test if the resampling logic produces non-zero values from mock data."""
    dataset = IMSPyTorchDataset(mock_m2aia_reader, mz_resolution=1.0)
    sample = dataset[0]
    
    # Since mock intensities are random, sum should be > 0
    assert torch.sum(sample) > 0

def test_exception_handling(mock_m2aia_reader):
    """Test if out-of-bounds access returns a zero tensor instead of crashing."""
    dataset = IMSPyTorchDataset(mock_m2aia_reader)
    # Index 99 is beyond our mocked 10 spectra
    sample = dataset[99]
    
    assert torch.all(sample == 0)
    assert sample.shape[0] == len(dataset.grid)