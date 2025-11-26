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

import warnings
from typing import Any, Optional

import numpy as np
import pyvista as pv
import vtk
from numcodecs import Blosc
from vtk.util import numpy_support

from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .constants import PhysicsConstants
from .domino_utils import (
    compute_face_area_numpy,
    compute_face_normal_numpy,
    decimate_mesh,
    get_volume_data,
    get_volume_point_data,
    to_float32,
)
from .schemas import (
    DoMINOExtractedDataInMemory,
    DoMINONumpyDataInMemory,
    DoMINONumpyMetadata,
    DoMINOZarrDataInMemory,
    PreparedZarrArrayInfo,
)


def extract_vtk_connectivity(mesh):
    """Extract raw connectivity arrays directly from VTK."""
    if hasattr(mesh, 'GetOutput'):
        ugrid = mesh.GetOutput()
    else:
        ugrid = mesh
    
    cells = ugrid.GetCells()
    cell_connectivity = numpy_support.vtk_to_numpy(cells.GetConnectivityArray())
    cell_offsets = numpy_support.vtk_to_numpy(cells.GetOffsetsArray())
    
    return cell_connectivity, cell_offsets, ugrid.GetNumberOfCells()


def build_face_connectivity_vtk(ugrid, points, cell_centers):
    """Build face connectivity and compute face geometry using pure VTK API.
    
    This function combines connectivity building with face geometry computation
    to avoid duplicate iteration over millions of faces.
    
    Args:
        ugrid: VTK unstructured grid
        points: Mesh points [num_points, 3]
        cell_centers: Cell centers [num_cells, 3]
    
    Returns:
        neighbors: List of neighbor cell IDs for each cell
        face_point_ids_map: Dict mapping (cell1_id, cell2_id) to face point IDs
        face_areas_map: Dict mapping (cell1_id, cell2_id) to face area
        face_normals_map: Dict mapping (cell1_id, cell2_id) to face normal [3]
        cell_point_ids: List of point IDs for each cell
    """
    n_cells = ugrid.GetNumberOfCells()
    
    # Extract VTK connectivity
    cell_connectivity, cell_offsets, _ = extract_vtk_connectivity(ugrid)
    
    # Build cell point IDs
    cell_point_ids = []
    for cell_idx in range(n_cells):
        start = cell_offsets[cell_idx]
        end = cell_offsets[cell_idx + 1]
        cell_point_ids.append(list(cell_connectivity[start:end]))
    
    # Extract faces using VTK API - hash-based matching for O(1) neighbor lookup
    face_to_cells = {}
    
    for cell_idx in range(n_cells):
        cell = ugrid.GetCell(cell_idx)
        n_faces = cell.GetNumberOfFaces()
        
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids_vtk = face.GetPointIds()
            n_face_pts = face_point_ids_vtk.GetNumberOfIds()
            
            # Extract face point IDs
            face_pts = [face_point_ids_vtk.GetId(i) for i in range(n_face_pts)]
            face_tuple = tuple(sorted(face_pts))
            
            if face_tuple not in face_to_cells:
                face_to_cells[face_tuple] = []
            face_to_cells[face_tuple].append((cell_idx, np.array(face_pts, dtype=np.int64)))
    
    # Build neighbors and compute face geometry from face matches
    neighbors = [[] for _ in range(n_cells)]
    face_point_ids_map = {}
    face_areas_map = {}
    face_normals_map = {}
    
    for face_tuple, cell_list in face_to_cells.items():
        if len(cell_list) == 2:  # Internal face shared by 2 cells
            (cell1_id, face_pts1), (cell2_id, face_pts2) = cell_list
            
            # Add neighbors
            neighbors[cell1_id].append(cell2_id)
            neighbors[cell2_id].append(cell1_id)
            
            # Store face point IDs
            face_point_ids_map[(cell1_id, cell2_id)] = face_pts1
            face_point_ids_map[(cell2_id, cell1_id)] = face_pts1
            
            # Compute face geometry (shared for both directions)
            face_area = compute_face_area_numpy(face_pts1, points)
            
            # Compute normals (outward from each cell - opposite directions)
            face_normal_1 = compute_face_normal_numpy(face_pts1, points, cell_centers[cell1_id])
            face_normal_2 = compute_face_normal_numpy(face_pts1, points, cell_centers[cell2_id])
            
            # Store geometry for both directions
            face_areas_map[(cell1_id, cell2_id)] = face_area
            face_areas_map[(cell2_id, cell1_id)] = face_area
            face_normals_map[(cell1_id, cell2_id)] = face_normal_1
            face_normals_map[(cell2_id, cell1_id)] = face_normal_2
    
    return neighbors, face_point_ids_map, face_areas_map, face_normals_map, cell_point_ids


class DoMINONumpyTransformation(DataTransformation):
    """Transforms DoMINO data for NumPy storage format (legacy support)."""

    def __init__(self, cfg: ProcessingConfig):
        super().__init__(cfg)

    def transform(self, data: DoMINOExtractedDataInMemory) -> DoMINONumpyDataInMemory:
        """Transform data for NumPy storage.

        Note: This is a legacy format with minimal metadata support.
        For full metadata support, use Zarr storage format instead.

        Args:
            data: DoMINO extracted data in memory

        Returns:
            Data formatted for NumPy storage with basic metadata
        """
        # Create minimal metadata
        numpy_metadata = DoMINONumpyMetadata(
            filename=data.metadata.filename,
            stream_velocity=data.metadata.stream_velocity,
            air_density=data.metadata.air_density,
        )

        return DoMINONumpyDataInMemory(
            stl_coordinates=to_float32(data.stl_coordinates),
            stl_centers=to_float32(data.stl_centers),
            stl_faces=to_float32(data.stl_faces),
            stl_areas=to_float32(data.stl_areas),
            metadata=numpy_metadata,
            surface_mesh_centers=to_float32(data.surface_mesh_centers),
            surface_normals=to_float32(data.surface_normals),
            surface_areas=to_float32(data.surface_areas),
            surface_fields=to_float32(data.surface_fields),
            volume_mesh_centers=to_float32(data.volume_mesh_centers),
            volume_fields=to_float32(data.volume_fields),
        )


class DoMINOPreprocessingTransformation(DataTransformation):
    """General preprocessing of data for DoMINO model."""

    DECIMATION_ALGOS: tuple[str, ...] = ("decimate_pro", "decimate")

    def __init__(
        self,
        cfg: ProcessingConfig,
        surface_variables: Optional[dict[str, str]] = None,
        volume_variables: Optional[dict[str, str]] = None,
        decimation: Optional[dict[str, Any]] = None,
    ):
        super().__init__(cfg)

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables

        self.decimation_algo = None
        self.target_reduction = None
        if decimation is not None:
            self.decimation_algo = decimation.get("algo")
            if self.decimation_algo not in self.DECIMATION_ALGOS:
                raise ValueError(
                    f"Unsupported decimation algo {self.decimation_algo}, must be one of {', '.join(self.DECIMATION_ALGOS)}"
                )
            self.target_reduction = decimation.get("reduction", 0.0)
            if not 0 <= self.target_reduction < 1.0:
                raise ValueError(
                    f"Expected value in [0, 1), got {self.target_reduction}"
                )
            # Copy decimation dict, excluding 'algo' and 'reduction'
            self.decimation_kwargs = {
                k: v for k, v in decimation.items() if k not in ("algo", "reduction")
            }

        self.constants = PhysicsConstants()

    def transform(
        self, data: DoMINOExtractedDataInMemory
    ) -> DoMINOExtractedDataInMemory:
        """Transform data for preprocessing."""

        # Process STL data
        mesh_stl = data.stl_polydata
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        # Delete raw STL data to save memory
        data.stl_polydata = None

        # Update processed STL data
        data.stl_coordinates = to_float32(stl_vertices)
        data.stl_centers = to_float32(stl_centers)
        data.stl_faces = to_float32(mesh_indices_flattened)
        data.stl_areas = to_float32(stl_sizes)

        # Update metadata
        bounds = mesh_stl.bounds
        data.metadata.x_bound = bounds[0:2]  # xmin, xmax
        data.metadata.y_bound = bounds[2:4]  # ymin, ymax
        data.metadata.z_bound = bounds[4:6]  # zmin, zmax
        data.metadata.num_points = len(mesh_stl.points)
        data.metadata.num_faces = len(mesh_indices_flattened)
        data.metadata.stream_velocity = self.constants.STREAM_VELOCITY
        data.metadata.air_density = self.constants.AIR_DENSITY

        # Load volume data if needed
        if data.volume_unstructured_grid is not None:
            # Compute cell centers once (used by both processing and connectivity)
            cell_centers_filter = vtk.vtkCellCenters()
            cell_centers_filter.SetInputData(data.volume_unstructured_grid)
            cell_centers_filter.Update()
            cell_centers = numpy_support.vtk_to_numpy(
                cell_centers_filter.GetOutput().GetPoints().GetData()
            ).astype(np.float64)
            
            # Process volume data (pass pre-computed cell centers)
            length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
            volume_fields_cell, volume_fields_point = self._process_volume_data(
                data.volume_unstructured_grid, length_scale, cell_centers
            )

            # Compute connectivity BEFORE deleting the unstructured grid (pass pre-computed cell centers)
            data = self._compute_connectivity(data, cell_centers)

            # Delete raw volume data to save memory
            data.volume_unstructured_grid = None

            # Update processed volume data
            data.volume_mesh_centers = to_float32(cell_centers)
            data.volume_fields = to_float32(volume_fields_cell)  # Existing: cell data
            data.volume_fields_point_data = to_float32(volume_fields_point)  # New: point data

        if data.surface_polydata is not None:

            # Process surface data
            (
                surface_coordinates,
                surface_normals,
                surface_sizes,
                surface_fields,
            ) = self._process_surface_data(data.surface_polydata)

            # Delete raw surface data to save memory
            data.surface_polydata = None

            # Update processed surface data
            data.surface_mesh_centers = to_float32(surface_coordinates)
            data.surface_normals = to_float32(surface_normals)
            data.surface_areas = to_float32(surface_sizes)
            data.surface_fields = to_float32(surface_fields)

            # Update metadata
            data.metadata.decimation_algo = self.decimation_algo
            data.metadata.decimation_reduction = self.target_reduction

        return data

    def _compute_connectivity(
        self, data: DoMINOExtractedDataInMemory, cell_centers: np.ndarray
    ) -> DoMINOExtractedDataInMemory:
        """Compute connectivity from volume mesh.
        
        Args:
            data: DoMINO extracted data
            cell_centers: Pre-computed cell centers [num_cells, 3] (avoids recomputation)
        """
        ugrid = data.volume_unstructured_grid
        
        # Extract points
        points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)
        
        # Compute cell volumes if not present
        if not ugrid.GetCellData().HasArray("Volume"):
            cell_size_filter = vtk.vtkCellSizeFilter()
            cell_size_filter.SetInputData(ugrid)
            cell_size_filter.SetComputeLength(False)
            cell_size_filter.SetComputeArea(False)
            cell_size_filter.SetComputeVolume(True)
            cell_size_filter.SetComputeVertexCount(False)
            cell_size_filter.Update()
            ugrid = cell_size_filter.GetOutput()
            data.volume_unstructured_grid = ugrid
        
        cell_volumes = numpy_support.vtk_to_numpy(
            ugrid.GetCellData().GetArray("Volume")
        ).astype(np.float64)
        
        # Build face connectivity and compute face geometry in one pass
        # Use pre-computed cell_centers (passed as argument to avoid redundant computation)
        neighbors, face_point_ids_map, face_areas_map, face_normals_map, cell_point_ids = build_face_connectivity_vtk(
            ugrid, points, cell_centers
        )
        
        # Flatten data structures
        cell_point_ids_flat = []
        cell_point_ids_offsets = [0]
        for cell_pts in cell_point_ids:
            cell_point_ids_flat.extend(cell_pts)
            cell_point_ids_offsets.append(len(cell_point_ids_flat))
        
        neighbors_flat = []
        neighbors_offsets = [0]
        face_point_ids_flat = []
        face_offsets = [0]
        
        # Lists for face geometry
        face_areas_flat = []
        face_normals_flat = []
        
        for cell_id, cell_neighbors in enumerate(neighbors):
            neighbors_flat.extend(cell_neighbors)
            neighbors_offsets.append(len(neighbors_flat))
            
            for neighbor_id in cell_neighbors:
                if (cell_id, neighbor_id) in face_point_ids_map:
                    face_pts = face_point_ids_map[(cell_id, neighbor_id)]
                    face_point_ids_flat.extend(face_pts)
                    
                    # Retrieve pre-computed face geometry
                    face_areas_flat.append(face_areas_map[(cell_id, neighbor_id)])
                    face_normals_flat.append(face_normals_map[(cell_id, neighbor_id)])
                    
                face_offsets.append(len(face_point_ids_flat))
        
        # Store connectivity
        data.volume_points = points.astype(np.float64)
        data.volume_cell_volumes = cell_volumes.astype(np.float64)
        data.volume_cell_centers = cell_centers.astype(np.float64)
        data.volume_cell_point_ids_flat = np.array(cell_point_ids_flat, dtype=np.int64)
        data.volume_cell_point_ids_offsets = np.array(cell_point_ids_offsets, dtype=np.int64)
        data.volume_neighbors_flat = np.array(neighbors_flat, dtype=np.int64)
        data.volume_neighbors_offsets = np.array(neighbors_offsets, dtype=np.int64)
        data.volume_face_point_ids_flat = np.array(face_point_ids_flat, dtype=np.int64)
        data.volume_face_offsets = np.array(face_offsets, dtype=np.int64)
        
        # Store face geometry
        data.volume_face_areas_flat = np.array(face_areas_flat, dtype=np.float64)
        data.volume_face_normals_flat = np.array(face_normals_flat, dtype=np.float64)  # Shape: [num_faces, 3]
        
        return data

    def _process_volume_data(
        self, unstructured_grid: vtk.vtkUnstructuredGrid, length_scale: float, cell_centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process volume mesh data, extracting cell data and point data.
        
        Args:
            unstructured_grid: VTK unstructured grid
            length_scale: Characteristic length scale for non-dimensionalization
            cell_centers: Pre-computed cell centers [num_cells, 3] (avoids recomputation)
        
        Returns:
            volume_fields_cell: Non-dimensionalized field values at cell centers
            volume_fields_point: Non-dimensionalized field values at nodes/vertices (or None)
        """

        # Extract cell data (existing behavior)
        cell_fields = get_volume_data(unstructured_grid, self.volume_variables, cell_centers)
        volume_fields_cell = np.concatenate(cell_fields, axis=-1)
        
        # Non-dimensionalize cell fields
        volume_fields_cell[:, :3] = volume_fields_cell[:, :3] / self.constants.STREAM_VELOCITY
        volume_fields_cell[:, 3:4] = volume_fields_cell[:, 3:4] / (
            self.constants.AIR_DENSITY * self.constants.STREAM_VELOCITY**2.0
        )
        volume_fields_cell[:, 4:] = volume_fields_cell[:, 4:] / (
            self.constants.STREAM_VELOCITY * length_scale
        )
        
        # Extract point data (new addition)
        point_fields = get_volume_point_data(unstructured_grid, self.volume_variables)
        volume_fields_point = None
        
        # Only process if we have point data available
        if any(f is not None for f in point_fields):
            valid_point_fields = [f for f in point_fields if f is not None]
            if valid_point_fields:
                volume_fields_point = np.concatenate(valid_point_fields, axis=-1)
                
                # Non-dimensionalize point fields
                volume_fields_point[:, :3] = volume_fields_point[:, :3] / self.constants.STREAM_VELOCITY
                volume_fields_point[:, 3:4] = volume_fields_point[:, 3:4] / (
                    self.constants.AIR_DENSITY * self.constants.STREAM_VELOCITY**2.0
                )
                volume_fields_point[:, 4:] = volume_fields_point[:, 4:] / (
                    self.constants.STREAM_VELOCITY * length_scale
                )

        return volume_fields_cell, volume_fields_point

    def _process_surface_data(
        self,
        mesh: pv.PolyData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process surface mesh data."""

        # Decimate mesh if needed.
        if self.decimation_algo is not None and self.target_reduction > 0:
            mesh = decimate_mesh(
                mesh,
                self.decimation_algo,
                self.target_reduction,
                self.decimation_kwargs,
            )

        cell_data = (mesh.cell_data[k] for k in self.surface_variables)
        surface_fields = np.concatenate(
            [d if d.ndim > 1 else d[:, np.newaxis] for d in cell_data], axis=-1
        )
        surface_coordinates = np.array(mesh.cell_centers().points)
        surface_normals = np.array(mesh.cell_normals)
        surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        surface_sizes = np.array(surface_sizes.cell_data["Area"])

        # Normalize cell normals
        surface_normals = (
            surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
        )

        # Non-dimensionalize surface fields
        surface_fields = surface_fields / (
            self.constants.AIR_DENSITY * self.constants.STREAM_VELOCITY**2.0
        )

        return surface_coordinates, surface_normals, surface_sizes, surface_fields


class DoMINOConnectivityTransformation(DataTransformation):
    """Computes volume mesh connectivity for FVM residual computation.
    
    This transformation extracts connectivity information from the volume mesh,
    including:
    - Cell point IDs and offsets
    - Cell neighbor connectivity
    - Face point IDs and offsets
    - Cell volumes and centers
    
    This connectivity data is required for computing physics-informed residuals
    using Finite Volume Method (FVM).
    """

    def __init__(self, cfg: ProcessingConfig, compute_connectivity: bool = True):
        super().__init__(cfg)
        self.compute_connectivity = compute_connectivity

    def transform(
        self, data: DoMINOExtractedDataInMemory
    ) -> DoMINOExtractedDataInMemory:
        """Compute and store connectivity data from volume mesh.
        
        Args:
            data: DoMINO extracted data containing volume_unstructured_grid
        
        Returns:
            Data with connectivity fields populated
        """
        if not self.compute_connectivity:
            return data

        # Check if volume data exists
        if data.volume_unstructured_grid is None:
            return data

        ugrid = data.volume_unstructured_grid
        
        # Extract points
        points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)
        
        # Compute cell volumes if not present
        if not ugrid.GetCellData().HasArray("Volume"):
            cell_size_filter = vtk.vtkCellSizeFilter()
            cell_size_filter.SetInputData(ugrid)
            cell_size_filter.SetComputeLength(False)
            cell_size_filter.SetComputeArea(False)
            cell_size_filter.SetComputeVolume(True)
            cell_size_filter.SetComputeVertexCount(False)
            cell_size_filter.Update()
            ugrid = cell_size_filter.GetOutput()
            # Update the reference in data
            data.volume_unstructured_grid = ugrid
        
        cell_volumes = numpy_support.vtk_to_numpy(
            ugrid.GetCellData().GetArray("Volume")
        ).astype(np.float64)
        
        # Compute cell centers
        cell_centers_filter = vtk.vtkCellCenters()
        cell_centers_filter.SetInputData(ugrid)
        cell_centers_filter.Update()
        cell_centers = numpy_support.vtk_to_numpy(
            cell_centers_filter.GetOutput().GetPoints().GetData()
        ).astype(np.float64)
        
        # Build face connectivity and compute face geometry in one pass
        neighbors, face_point_ids_map, face_areas_map, face_normals_map, cell_point_ids = build_face_connectivity_vtk(
            ugrid, points, cell_centers
        )
        
        # Flatten data structures for efficient storage and computation
        cell_point_ids_flat = []
        cell_point_ids_offsets = [0]
        for cell_pts in cell_point_ids:
            cell_point_ids_flat.extend(cell_pts)
            cell_point_ids_offsets.append(len(cell_point_ids_flat))
        
        neighbors_flat = []
        neighbors_offsets = [0]
        face_point_ids_flat = []
        face_offsets = [0]
        
        # Lists for face geometry
        face_areas_flat = []
        face_normals_flat = []
        
        for cell_id, cell_neighbors in enumerate(neighbors):
            neighbors_flat.extend(cell_neighbors)
            neighbors_offsets.append(len(neighbors_flat))
            
            for neighbor_id in cell_neighbors:
                if (cell_id, neighbor_id) in face_point_ids_map:
                    face_pts = face_point_ids_map[(cell_id, neighbor_id)]
                    face_point_ids_flat.extend(face_pts)
                    
                    # Retrieve pre-computed face geometry
                    face_areas_flat.append(face_areas_map[(cell_id, neighbor_id)])
                    face_normals_flat.append(face_normals_map[(cell_id, neighbor_id)])
                    
                face_offsets.append(len(face_point_ids_flat))
        
        # Convert to numpy arrays with appropriate dtypes
        data.volume_points = points.astype(np.float64)
        data.volume_cell_volumes = cell_volumes.astype(np.float64)
        data.volume_cell_centers = cell_centers.astype(np.float64)
        data.volume_cell_point_ids_flat = np.array(cell_point_ids_flat, dtype=np.int64)
        data.volume_cell_point_ids_offsets = np.array(cell_point_ids_offsets, dtype=np.int64)
        data.volume_neighbors_flat = np.array(neighbors_flat, dtype=np.int64)
        data.volume_neighbors_offsets = np.array(neighbors_offsets, dtype=np.int64)
        data.volume_face_point_ids_flat = np.array(face_point_ids_flat, dtype=np.int64)
        data.volume_face_offsets = np.array(face_offsets, dtype=np.int64)
        
        # Store face geometry
        data.volume_face_areas_flat = np.array(face_areas_flat, dtype=np.float64)
        data.volume_face_normals_flat = np.array(face_normals_flat, dtype=np.float64)  # Shape: [num_faces, 3]
        
        return data


class DoMINOZarrTransformation(DataTransformation):
    """Transforms DoMINO data for Zarr storage format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        compression_method: str = "zstd",
        compression_level: int = 5,
        chunk_size_mb: float = 1.0,  # Default 1MB chunk size
    ):
        super().__init__(cfg)
        self.compressor = Blosc(
            cname=compression_method,
            clevel=compression_level,
            shuffle=Blosc.SHUFFLE,
        )
        self.chunk_size_mb = chunk_size_mb

        # Warn if chunk size might be problematic
        if chunk_size_mb < 1.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too small. "
                "This could lead to poor performance due to overhead.",
                UserWarning,
            )
        elif chunk_size_mb > 50.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too large. "
                "This could lead to memory issues and poor random access performance.",
                UserWarning,
            )

    def _prepare_array(self, array: np.ndarray) -> PreparedZarrArrayInfo:
        """Prepare array for Zarr storage with compression and chunking."""
        if array is None:
            return None

        # Calculate chunk size based on configured size in MB
        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)  # Convert MB to bytes
        item_size = array.itemsize
        shape = array.shape

        if len(shape) == 1:
            chunk_size = min(shape[0], target_chunk_size // item_size)
            chunks = (chunk_size,)
        else:
            # For 2D arrays, try to keep rows together
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            chunks = (chunk_rows, shape[1])

        # Preserve dtype for integer arrays (connectivity), convert to float32 for others
        if np.issubdtype(array.dtype, np.integer):
            data = array  # Keep integer dtype as-is
        else:
            data = np.float32(array)  # Convert float arrays to float32

        return PreparedZarrArrayInfo(
            data=data,
            chunks=chunks,
            compressor=self.compressor,
        )

    def transform(self, data: DoMINOExtractedDataInMemory) -> DoMINOZarrDataInMemory:
        """Transform data for Zarr storage.

        Organizes data into hierarchical groups and applies compression settings.

        Args:
            data: Dictionary containing DoMINO data

        Returns:
            Dictionary with data formatted for Zarr storage, including:
                - Data organized into groups (stl, surface, volume, connectivity)
                - Compression settings
                - Chunking configurations
        """
        return DoMINOZarrDataInMemory(
            stl_coordinates=self._prepare_array(data.stl_coordinates),
            stl_centers=self._prepare_array(data.stl_centers),
            stl_faces=self._prepare_array(data.stl_faces),
            stl_areas=self._prepare_array(data.stl_areas),
            metadata=data.metadata,
            surface_mesh_centers=self._prepare_array(data.surface_mesh_centers),
            surface_normals=self._prepare_array(data.surface_normals),
            surface_areas=self._prepare_array(data.surface_areas),
            surface_fields=self._prepare_array(data.surface_fields),
            volume_mesh_centers=self._prepare_array(data.volume_mesh_centers),
            volume_fields=self._prepare_array(data.volume_fields),
            volume_fields_point_data=self._prepare_array(data.volume_fields_point_data),
            # Connectivity data
            volume_points=self._prepare_array(data.volume_points),
            volume_cell_volumes=self._prepare_array(data.volume_cell_volumes),
            volume_cell_centers=self._prepare_array(data.volume_cell_centers),
            volume_cell_point_ids_flat=self._prepare_array(data.volume_cell_point_ids_flat),
            volume_cell_point_ids_offsets=self._prepare_array(data.volume_cell_point_ids_offsets),
            volume_neighbors_flat=self._prepare_array(data.volume_neighbors_flat),
            volume_neighbors_offsets=self._prepare_array(data.volume_neighbors_offsets),
            volume_face_point_ids_flat=self._prepare_array(data.volume_face_point_ids_flat),
            volume_face_offsets=self._prepare_array(data.volume_face_offsets),
            # Face geometry
            volume_face_areas_flat=self._prepare_array(data.volume_face_areas_flat),
            volume_face_normals_flat=self._prepare_array(data.volume_face_normals_flat),
        )
