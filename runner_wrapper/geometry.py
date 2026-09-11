"""Undo optimization normalization and the panorama roll before publishing."""
from __future__ import annotations

import json
from pathlib import Path


def export_scene(source: Path, destination: Path, normalization: Path, angle: float) -> dict:
    import numpy as np
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    transform = json.loads(normalization.read_text())
    scale = float(transform["scale"])
    center = np.asarray(transform["center"], dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0 or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid reconstruction normalization")
    ply = PlyData.read(source)
    vertices = ply["vertex"].data
    required = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    if not len(vertices) or not set(required).issubset(vertices.dtype.names or ()):
        raise ValueError("Expected a non-empty 3D Gaussian PLY, not a plain point cloud")
    if any(not np.isfinite(vertices[name]).all() for name in required):
        raise ValueError("3DGS contains non-finite values")
    # Match upstream's integer pixel roll on its 2048-wide conditioning panorama.
    shift = (int(angle / 360 * 2048 + 1024) % 2048 + 1024) % 2048
    rotation = Rotation.from_euler("y", shift * 360 / 2048, degrees=True)
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]) * scale + center
    points = rotation.apply(points)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for name in ("scale_0", "scale_1", "scale_2"):
        vertices[name] += np.log(scale)
    # Gaussian PLY stores scalar-first quaternions; scipy takes scalar-last.
    quats = np.column_stack([vertices[name] for name in ("rot_1", "rot_2", "rot_3", "rot_0")])
    rotated = (rotation * Rotation.from_quat(quats)).as_quat()
    for index, name in enumerate(("rot_1", "rot_2", "rot_3", "rot_0")):
        vertices[name] = rotated[:, index]
    if all(name in vertices.dtype.names for name in ("nx", "ny", "nz")):
        normals = rotation.apply(np.column_stack([vertices[name] for name in ("nx", "ny", "nz")]))
        for index, name in enumerate(("nx", "ny", "nz")):
            vertices[name] = normals[:, index]
    ply.write(destination)
    return {"scene_coordinate_system": "RDF", "scene_units": "relative",
            "scene_origin": "primary_viewpoint", "scale_calibration_required": True}
