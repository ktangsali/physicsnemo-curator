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

from examples.external_aerodynamics.external_aero_utils import to_float32
from examples.external_aerodynamics.schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
)


def default_geometry_processing_for_external_aerodynamics(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Default geometry processing for External Aerodynamics."""

    data.stl_coordinates = data.stl_polydata.points
    
    # Handle both faces (3D surfaces) and lines (2D profiles)
    if data.stl_polydata.faces is not None and len(data.stl_polydata.faces) > 0:
        # 3D surface with triangular faces
        data.stl_faces = (
            np.array(data.stl_polydata.faces).reshape((-1, 4))[:, 1:].astype(np.int32)
        ).flatten()  # Assuming triangular elements
        # Compute areas for 3D faces
        data.stl_areas = data.stl_polydata.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        data.stl_areas = np.array(data.stl_areas.cell_data["Area"])
    elif data.stl_polydata.lines is not None and len(data.stl_polydata.lines) > 0:
        # 2D profile with line segments - extract connectivity
        lines = data.stl_polydata.lines
        # Lines format: [n_pts, pt1, pt2, n_pts, pt1, pt2, ...]
        # Extract pairs for line segments (each segment has 2 points)
        faces_list = []
        i = 0
        while i < len(lines):
            n_pts = lines[i]
            pts = lines[i+1:i+1+n_pts]
            faces_list.extend(pts)
            i += 1 + n_pts
        data.stl_faces = np.array(faces_list, dtype=np.int32)
        # Compute line lengths for 2D lines
        data.stl_areas = data.stl_polydata.compute_cell_sizes(
            length=True, area=False, volume=False
        )
        data.stl_areas = np.array(data.stl_areas.cell_data["Length"])
    else:
        # No faces or lines - empty array
        data.stl_faces = np.array([], dtype=np.int32)
        data.stl_areas = np.array([], dtype=np.float32)
    
    data.stl_centers = np.array(data.stl_polydata.cell_centers().points)

    # Update metadata
    bounds = data.stl_polydata.bounds
    data.metadata.x_bound = bounds[0:2]  # xmin, xmax
    data.metadata.y_bound = bounds[2:4]  # ymin, ymax
    data.metadata.z_bound = bounds[4:6]  # zmin, zmax
    data.metadata.num_points = len(data.stl_polydata.points)
    data.metadata.num_faces = len(data.stl_faces)

    return data


def update_geometry_data_to_float32(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """Update geometry data to float32."""

    data.stl_coordinates = to_float32(data.stl_coordinates)
    data.stl_centers = to_float32(data.stl_centers)
    data.stl_areas = to_float32(data.stl_areas)
    # data.stl_faces will be left as is (np.int32)

    return data
