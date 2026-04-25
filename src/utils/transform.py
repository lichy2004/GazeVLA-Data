import numpy as np


def create_rotation_transform(axis, angle_degrees):
    rotation_matrix = create_rotation_matrix(axis, angle_degrees)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix
    return transform


def create_rotation_matrix(axis, angle_degrees):
    if axis == "x":
        return create_rotation_matrix_x(angle_degrees)
    if axis == "y":
        return create_rotation_matrix_y(angle_degrees)
    if axis == "z":
        return create_rotation_matrix_z(angle_degrees)
    raise ValueError(f"Invalid axis: {axis}")


def create_rotation_matrix_x(angle_degrees):
    """Rotate clockwise around the X axis."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array(
        [
            [1, 0, 0],
            [0, cos_a, sin_a],
            [0, -sin_a, cos_a],
        ],
        dtype=np.float32,
    )


def create_rotation_matrix_y(angle_degrees):
    """Rotate clockwise around the Y axis."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array(
        [
            [cos_a, 0, -sin_a],
            [0, 1, 0],
            [sin_a, 0, cos_a],
        ],
        dtype=np.float32,
    )


def create_rotation_matrix_z(angle_degrees):
    """Rotate clockwise around the Z axis."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array(
        [
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
