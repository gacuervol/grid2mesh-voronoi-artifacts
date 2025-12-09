import pytest 
import numpy as np
import xarray as xr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from download_data import generate_grid_weights_from_latitude_cosine


@pytest.fixture
def latitudes():
    return np.array([-90, -45, 0, 45, 90])

@pytest.fixture
def longitudes():
    return np.array([0, 90, 180])

@pytest.fixture
def sea_mask_2d(latitudes, longitudes):
    mask_data = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 0, 1]])
    
    return xr.DataArray(
        data=mask_data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": latitudes[:3], 
            "longitude": longitudes
            }
    ), mask_data

@pytest.fixture
def sea_mask_3d(latitudes, longitudes):
    mask_data_3d = np.array([[[0, 1, 0],
                            [1, 1, 1],
                            [0, 0, 1]]])
    return xr.DataArray(
        data=mask_data_3d,
        dims=["depth", "latitude", "longitude"],
        coords={
            "depth": [0],
            "latitude": latitudes[:3],
            "longitude": longitudes
        }
    ), mask_data_3d[0]

@pytest.mark.parametrize("mask_fixture", ["sea_mask_2d", "sea_mask_3d"])
class TestGenerateGridWeightsFromLatitudeCosine():

    def test_result_have_2_dimensions(self, mask_fixture, request):
        """ Test that the result has 2 dimensions """
        mask, _ = request.getfixturevalue(mask_fixture)
        result = generate_grid_weights_from_latitude_cosine(mask)

        assert len(result.shape) == 2


    def test_result_shape_matches_mask_shape(self, mask_fixture:xr.DataArray, request):
        """ Test that the result shape matches the sea surface mask shape """
        mask, mask_data = request.getfixturevalue(mask_fixture)
        result = generate_grid_weights_from_latitude_cosine(mask)

        assert result.shape == mask_data.shape

    
    def test_land_positions_have_zero_weight(self, mask_fixture:xr.DataArray, request):
        """ Test that the land positions have zero grid weight """
        mask, mask_data = request.getfixturevalue(mask_fixture)
        result = generate_grid_weights_from_latitude_cosine(mask)

        land_positions = mask_data == 0
        land_weights = result[land_positions]
        assert np.all(land_weights == 0)

    def test_ocean_positions_have_positive_weight(self, mask_fixture:xr.DataArray, request):
        """ Test that the ocean positions have positive grid weight """
        mask, mask_data = request.getfixturevalue(mask_fixture)
        result = generate_grid_weights_from_latitude_cosine(mask)

        ocean_positions = mask_data == 1
        ocean_weights = result[ocean_positions]
        assert np.all(ocean_weights > 0)
    
    def test_weights_are_normalized(self, mask_fixture:xr.DataArray, request):
        """ Test that the weights are normalized """
        mask, mask_data = request.getfixturevalue(mask_fixture)
        result = generate_grid_weights_from_latitude_cosine(mask)

        ocean_positions = mask_data == 1
        ocean_weights = result[ocean_positions]
        assert np.isclose(np.mean(ocean_weights), 1.0, atol=1e-6)






    