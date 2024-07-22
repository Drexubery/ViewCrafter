import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import roma

def interpolate_poses(R_matrices, T_vectors, num_interpolations, intervals=1):
    """
    Interpolate between a series of rotation matrices and translation vectors.
    
    Parameters:
    - R_matrices: A list of rotation matrices, each with shape (3, 3).
    - T_vectors: A list of translation vectors, each with shape (3,).
    - num_interpolations: The total number of poses to generate, including the start and end poses.
    
    Returns:
    - interpolated_Rs: A list of interpolated rotation matrices.
    - interpolated_Ts: A list of interpolated translation vectors.
    """
    # Validate input shapes
    for R_mat in R_matrices:
        assert R_mat.shape == (3, 3), f"Invalid rotation matrix shape: {R_mat.shape}"
    for T_vec in T_vectors:
        assert T_vec.shape == (3,), f"Invalid translation vector shape: {T_vec.shape}"

    # Convert rotation matrices to quaternions
    quaternions = [R.from_matrix(R_mat).as_quat() for R_mat in R_matrices]
    quaternions = np.array(quaternions)
    T_vectors = np.array(T_vectors)
    
    quaternions = quaternions[::intervals]
    T_vectors = T_vectors[::intervals]

    # Create interpolation time points
    times = np.linspace(0, 1, num=len(R_matrices) // intervals)
    target_times = np.linspace(0, 1, num=num_interpolations // intervals)
    
    # Interpolate quaternions and translation vectors
    # kind : { linear, quadratic, cubic, nearest, zero, slinear}
    quat_interpolator = interp1d(times, quaternions, axis=0, kind='linear')
    T_interpolator = interp1d(times, T_vectors, axis=0, kind='linear')
    
    interpolated_quats = quat_interpolator(target_times)
    interpolated_Ts = T_interpolator(target_times)
    
    # Convert interpolated quaternions back to rotation matrices
    interpolated_Rs = [R.from_quat(quat).as_matrix() for quat in interpolated_quats]
    
    return interpolated_Rs, interpolated_Ts

# Example usage

