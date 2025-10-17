# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pyvista as pv

from examples.external_aerodynamics.constants import PhysicsConstants
from examples.external_aerodynamics.external_aero_utils import (
    get_volume_data,
    to_float32,
)
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)


def default_volume_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
    volume_variables: list[str],
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default volume processing for External Aerodynamics."""
    data.volume_mesh_centers, data.volume_fields = get_volume_data(
        data.volume_unstructured_grid, volume_variables
    )
    data.volume_fields = np.concatenate(data.volume_fields, axis=-1)
    return data


def non_dimensionalize_volume_fields(
    data: ExternalAerodynamicsExtractedDataInMemory,
    air_density: float = PhysicsConstants.AIR_DENSITY,
    stream_velocity: float = PhysicsConstants.STREAM_VELOCITY,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Non-dimensionalize volume fields."""

    stl_vertices = data.stl_polydata.points
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
    data.volume_fields[:, :3] = data.volume_fields[:, :3] / stream_velocity
    data.volume_fields[:, 3:4] = data.volume_fields[:, 3:4] / (
        air_density * stream_velocity**2.0
    )
    data.volume_fields[:, 4:] = data.volume_fields[:, 4:] / (
        stream_velocity * length_scale
    )
    return data


def update_volume_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update volume data to float32."""
    data.volume_mesh_centers = to_float32(data.volume_mesh_centers)
    data.volume_fields = to_float32(data.volume_fields)
    if data.volume_sdf is not None:
        data.volume_sdf = to_float32(data.volume_sdf)
    return data


def read_implicit_distance_as_sdf(
    data: ExternalAerodynamicsExtractedDataInMemory,
    field_name: str = "implicit_distance",
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Read pre-computed implicit distance from volume data and store as volume_sdf.
    
    This reads the implicit distance field that already exists in the VTU file
    (computed during the simulation or preprocessing) and stores it as volume_sdf.
    
    Args:
        data: ExternalAerodynamicsExtractedDataInMemory with volume data
        field_name: Name of the implicit distance field in the volume data
        
    Returns:
        Data with volume_sdf populated from existing field
    """
    if data.volume_unstructured_grid is None:
        return data
    
    volume = pv.wrap(data.volume_unstructured_grid)
    
    # Check if field exists in point data or cell data
    if field_name in volume.point_data:
        sdf = np.array(volume.point_data[field_name])
    elif field_name in volume.cell_data:
        sdf = np.array(volume.cell_data[field_name])
    else:
        raise ValueError(
            f"Field '{field_name}' not found in volume data. "
            f"Available point data: {list(volume.point_data.keys())}. "
            f"Available cell data: {list(volume.cell_data.keys())}"
        )
    
    # Reshape to (N, 1) if needed
    if sdf.ndim == 1:
        sdf = sdf.reshape(-1, 1)
    
    data.volume_sdf = sdf.astype(np.float32)
    
    return data


def clip_volume_to_box(
    data: ExternalAerodynamicsExtractedDataInMemory,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-2, 4), (-1.5, 1.5), (0, 1)
    ),
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Clip volume data to a bounding box.
    
    Args:
        data: ExternalAerodynamicsExtractedDataInMemory with volume data
        bounds: Tuple of (x_bounds, y_bounds, z_bounds) where each is (min, max)
        
    Returns:
        Data with clipped volume grid
    """
    if data.volume_unstructured_grid is None:
        return data
    
    volume = pv.wrap(data.volume_unstructured_grid)
    
    x_bounds, y_bounds, z_bounds = bounds
    box_bounds = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]]
    
    clipped = volume.clip_box(box_bounds, invert=False, crinkle=True)
    
    data.volume_unstructured_grid = clipped
    
    return data
