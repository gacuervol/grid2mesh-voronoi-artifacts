import os
import pytest
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from download_data import download_static_atlantic

@pytest.fixture(scope="session")
def generate_files_once(tmpdir_factory):
    path_prefix = str(tmpdir_factory.mktemp("data"))
    download_static_atlantic(path_prefix)
    
    return path_prefix

class TestDownloadStaticAtlantic():
    def test_bathy_mask_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "bathy_mask.nc")
        assert os.path.exists(file_path), "Archivo 'bathy_mask.nc' no encontrado."

    def test_sea_mask_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "sea_mask.npy")
        assert os.path.exists(file_path), "Archivo 'sea_mask.npy' no encontrado."
        
        data = np.load(file_path)
        assert data.size > 0, "El archivo 'sea_mask.npy' está vacío."

    def test_sea_depth_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "sea_depth.npy")
        assert os.path.exists(file_path), "Archivo 'sea_depth.npy' no encontrado."
        
        data = np.load(file_path)
        assert data.size > 0, "El archivo 'sea_depth.npy' está vacío."

    def test_boundary_mask_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "boundary_mask.npy")
        assert os.path.exists(file_path), "Archivo 'boundary_mask.npy' no encontrado."
        
        data = np.load(file_path)
        assert data.size > 0, "El archivo 'boundary_mask.npy' está vacío."

    def test_coordinates_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "coordinates.npy")
        assert os.path.exists(file_path), "Archivo 'coordinates.npy' no encontrado."

    def test_nwp_xy_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "nwp_xy.npy")
        assert os.path.exists(file_path), "Archivo 'nwp_xy.npy' no encontrado."

    def test_grid_weights_exists(self, generate_files_once):
        file_path = os.path.join(generate_files_once, "grid_weights.npy")
        assert os.path.exists(file_path), "Archivo 'grid_weights.npy' no encontrado."
