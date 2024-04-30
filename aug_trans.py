import math
import torch
import random
import numpy as np
import torch.nn.functional as F


def rot_matrix(alpha, u, v):
    """
    Generates an N-dimensional rotation matrix based on Rodrigues' rotation formula.
    
    Parameters:
    alpha (float): Angle of rotation in radians (counter-clockwise).
    u, v (torch.Tensor): Vectors defining the plane of rotation.
    
    Returns:
    torch.Tensor: Rotation matrix.
    """
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    
    # Normalize vectors
    u = u / torch.linalg.norm(u)
    v = v - torch.mm(u.t(), v) * u  # Make v orthogonal to u
    v = v / torch.linalg.norm(v)
    
    # Rodrigues' rotation formula
    n = u.size(0)
    R = torch.eye(n) + s * (torch.mm(v, u.t()) - torch.mm(u, v.t())) + (c - 1) * (torch.mm(u, u.t()) + torch.mm(v, v.t()))
    
    return R


def rotate_matrix(matrix, angle):
    """
    Function to rotate the matrix by the specified angle (in degrees).
    
    Parameters:
    matrix (torch.Tensor): The input matrix to rotate.
    angle (float): The rotation angle in degrees.
    
    Returns:
    torch.Tensor: The rotated matrix.
    """
    # Convert angle to radians
    angle_radians = torch.tensor(math.radians(angle))
    
    # Generate random vectors u and v for N-dimensional rotation
    u = torch.randn(matrix.size(0), 1, dtype=matrix.dtype)
    v = torch.randn(matrix.size(0), 1, dtype=matrix.dtype)
    
    # Generate the rotation matrix
    R = rot_matrix(angle_radians, u, v)
    
    # Rotate the matrix using matrix multiplication
    rotated_matrix = torch.mm(R, matrix)
    
    return rotated_matrix


def cutout_matrix(matrix, n_elements_to_keep):
    """
    Function to remove random elements from the matrix.

    Parameters:
    matrix (torch.Tensor): The input matrix from which elements will be removed.
    num_elements_to_remove (int): The number of elements to remove.

    Returns:
    torch.Tensor: The matrix after removing the specified number of elements.
    torch.Tensor: The indices of the elements that were kept.
    """
    # Generate a random permutation of indices
    rp = torch.randperm(matrix.size(0))
    
    # Select indices to keep
    indices_to_keep = rp[:n_elements_to_keep]
    
    # Create the cutout matrix by indexing the original matrix with the selected indices
    cutout_matrix = matrix[indices_to_keep, :]
    
    return cutout_matrix, indices_to_keep


def add_gaussian_noise(matrix, noise_std):
    """
    Function to add Gaussian noise to the matrix.

    Parameters:
    matrix (torch.Tensor): The input matrix to which noise will be added.
    noise_level (float): The level of noise to add.

    Returns:
    torch.Tensor: The noisy matrix.
    """
    # Generate Gaussian noise with mean 0 and the specified standard deviation
    # The noise matrix is of the same size as the original matrix
    noise = noise_std * torch.randn(matrix.shape)
    
    # Add the noise to the original matrix
    noisy_matrix = matrix + noise
    
    return noisy_matrix


def apply_gaussian_blur(matrix, sigma, kernel_size=3):
    """
    Function to apply Gaussian blur to the columns of the matrix.
    
    Args:
    - matrix (torch.Tensor): Input matrix of shape (rows, cols).
    - sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
    - torch.Tensor: Blurred matrix of shape (rows, cols).
    """
    # If sigma is 0, return the same matrix
    if sigma == 0:
        return matrix
    
    rows, cols = matrix.shape

    # Generate Gaussian kernel
    x = torch.linspace(-kernel_size / 2, kernel_size / 2, kernel_size, dtype=matrix.dtype)
    kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Kernel normalization

    # Adjust kernel for 1D convolution and groups: (cols, 1, kernel_size)
    # Each column (now treated as a separate group) will use its own kernel slice for convolution
    kernel = kernel.repeat(cols, 1, 1)

    # Transpose matrix and add batch dimension: [1, cols, rows], treating each column as a separate channel
    matrix_with_channel = matrix.t().unsqueeze(0)

    # Apply Gaussian blur to each column independently
    blurred_matrix = F.conv1d(matrix_with_channel, kernel, padding=kernel_size//2, groups=cols)

    # Transpose back to original shape and remove batch dimension
    blurred_matrix = blurred_matrix.squeeze(0).t()

    return blurred_matrix.squeeze()


def apply_aug_trans(matrix, transformations, max_angle=5, cutout_perc=0.4, max_noise=1, max_blur=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    # Convert input matrix to tensor if it's not already one
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float64)
    
    augmented_matrix = matrix.clone()
    indices_to_keep = torch.arange(matrix.size(0))
    
    for transformation in transformations:
        if transformation == 'rotate':
            angle = (random.random() * (2 * max_angle)) - max_angle
            augmented_matrix = rotate_matrix(augmented_matrix, angle)
        elif transformation == 'cutout':
            n_elements_to_keep = round((1 - cutout_perc) * matrix.size(0))
            _, indices_to_keep = cutout_matrix(augmented_matrix, n_elements_to_keep)
        elif transformation == 'gaussian_noise':
            noise_level = random.random() * max_noise
            augmented_matrix = add_gaussian_noise(augmented_matrix, noise_level)
        elif transformation == 'gaussian_blur':
            sigma = random.random() * max_blur
            augmented_matrix = apply_gaussian_blur(augmented_matrix, sigma)
        else:
            raise ValueError(f'Invalid transformation: {transformation}')
    
    return augmented_matrix, indices_to_keep
