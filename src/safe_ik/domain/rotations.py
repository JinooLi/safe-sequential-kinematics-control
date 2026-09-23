"""Backend-independent rotation and quaternion operations."""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized scalar-first quaternion."""
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def quaternion_orientation_error_components(
    desired_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return scalar/vector parts of ``Q_desired * inverse(Q_current)``.

    The desired quaternion is flipped to the current hemisphere so the error
    is invariant to the two quaternion representations of the same rotation.
    """
    desired = np.asarray(desired_wxyz, dtype=float).copy()
    current = np.asarray(current_wxyz, dtype=float)
    if desired.shape != (4,) or current.shape != (4,):
        raise ValueError("quaternions must have shape (4,) in [w, x, y, z] order")
    desired /= np.linalg.norm(desired)
    current /= np.linalg.norm(current)
    if desired @ current < 0.0:
        desired = -desired
    eta_d, eta_e = desired[0], current[0]
    eps_d, eps_e = desired[1:], current[1:]
    scalar = float(eta_d * eta_e + eps_d @ eps_e)
    vector = eta_e * eps_d - eta_d * eps_e - np.cross(eps_d, eps_e)
    return scalar, vector


def quaternion_orientation_error(
    desired_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> np.ndarray:
    """Return the vector part of ``Q_desired * inverse(Q_current)``."""
    return quaternion_orientation_error_components(desired_wxyz, current_wxyz)[1]
