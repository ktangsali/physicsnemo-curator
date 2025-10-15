# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyvista as pv

from .schemas import ExternalAerodynamicsExtractedDataInMemory


def compute_sdf(
    data: ExternalAerodynamicsExtractedDataInMemory,
) -> ExternalAerodynamicsExtractedDataInMemory:
    """
    Compute signed distance field from surface boundary to volume points.
    
    Uses the surface polydata (aerofoil) as the boundary and computes
    the signed distance for all volume mesh centers.
    
    Args:
        data: ExternalAerodynamicsExtractedDataInMemory with surface and volume data
        
    Returns:
        Updated data with sdf field added to volume_sdf
    """
    if data.surface_polydata is None or data.volume_mesh_centers is None:
        return data
    
    surface = data.surface_polydata
    volume_centers = data.volume_mesh_centers
    
    surface_mesh = pv.PolyData(surface.points, surface.faces)
    
    implicit_distance = pv.PolyData(volume_centers).compute_implicit_distance(
        surface_mesh, inplace=False
    )
    
    sdf = -implicit_distance['implicit_distance']
    data.volume_sdf = sdf.reshape(-1, 1).astype(np.float32)
    
    return data

