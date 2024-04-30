import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dictlearn import omp
from aug_trans import apply_aug_trans


def maximize_class_assignments(usage_stats, n_atoms_per_class, n_classes):
    # Ensure the matrix is (n*c, c)
    assert usage_stats.shape == (n_atoms_per_class * n_classes, n_classes)
    
    # Create an expanded cost matrix for the linear sum assignment algorithm
    # Multiply by -1 to maximize using a min-cost algorithm
    cost_matrix = np.tile(-usage_stats, (1, n_atoms_per_class))

    # Run the linear sum assignment algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Extract the results
    assignment = col_ind % n_classes

    # Compute the total probability (maximized sum)
    max_probability = usage_stats[row_ind, assignment].sum()

    return assignment, max_probability


def raw_class_assignments(usage_stats, n_atoms_per_class, n_classes):
    atom_classes = -1 * np.ones(n_atoms_per_class*n_classes, dtype=int)
    available_atoms = set(range(n_atoms_per_class*n_classes))
    
    # Sort atoms for each class based on decreasing frequency of use
    for class_idx in range(n_classes):
        class_atoms_sorted = np.argsort(-usage_stats[:, class_idx])
        
        assigned_count = 0
        for atom_idx in class_atoms_sorted:
            if atom_idx in available_atoms and assigned_count < n_atoms_per_class:
                atom_classes[atom_idx] = class_idx
                available_atoms.remove(atom_idx)
                assigned_count += 1
            
            if assigned_count >= n_atoms_per_class:
                break
    
    return atom_classes


def atom_distribution(Y, D, n_nonzero_coefs, n_classes):
    n_samples_per_class = Y.shape[1] // n_classes
    n_atoms_per_class = D.shape[1] // n_classes
   
    # Step 1: Run OMP for each sample and gather atom usage statistics
    usage_stats = np.zeros((D.shape[1], n_classes))
    
    for class_idx in range(n_classes):
        class_start = class_idx * n_samples_per_class
        class_end = class_start + n_samples_per_class
        Y_class = Y[:, class_start:class_end]
        
        X_class = omp(Y_class, D, n_nonzero_coefs)
        for i in np.nonzero(X_class)[:, 0]:
            usage_stats[i, class_idx] += 1

    # Step 2: Assign atoms to classes ensuring each class gets exactly m atoms
    atom_classes = raw_class_assignments(usage_stats, n_atoms_per_class, n_classes)
    # atom_classes, _ = maximize_class_assignments(usage_stats, n_atoms_per_class, n_classes)

    return atom_classes


def simclr_loss(encodings, temperature=0.5):
    """
    Computes the SimCLR loss given a batch of embeddings.
    
    Args:
    - X (torch.Tensor): The input tensor containing embeddings of shape (2N, D),
                        where D is the dimensionality of the embeddings.
    - temperature (float): The temperature parameter for scaling.
    
    Returns:
    - torch.Tensor: The SimCLR loss.
    """
    n_views = 2
    device = encodings.device
    batch_size = encodings.shape[1] // 2

    # Normalize the embeddings
    encodings = F.normalize(encodings, dim=0)

    # Compute cosine similarity
    similarity_matrix = torch.matmul(encodings.T, encodings)

    # Create mask to select positive samples and ignore diagonal elements
    mask = torch.eye(batch_size * 2, device=device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels = labels.fill_diagonal_(0)

    # Extract positive similarities (similarity of each sample with its positive pair)
    positives = similarity_matrix[labels].view(2 * batch_size, 1)

    # Extract negative similarities
    negatives = similarity_matrix[~labels].view(2 * batch_size, -1)

    # Compute logits
    logits = torch.cat((positives, negatives), dim=1)
    logits /= temperature

    # Compute labels: positives are the first one
    labels = torch.zeros(2 * batch_size, device=device, dtype=torch.long)

    # Compute the cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def optimize_dictionary(D, X_train, n_nonzero_coefs, batch_size, n_iterations, learning_rate, transformations, max_angle, cutout_perc, noise_std, blur_std):
    n_features = X_train.shape[0]

    # Initialize an optimizer
    optimizer = optim.SGD([D], lr=learning_rate)

    for i_iter in tqdm(range(n_iterations)):
        random_permutation = torch.randperm(X_train.shape[1])
        train_idxs = random_permutation[:batch_size]

        # Prepare features for augmentation
        features = torch.cat((X_train[:, train_idxs], X_train[:, train_idxs]), dim=1)

        # Extract features using augmentation transformations
        Y, indices_to_keep = apply_aug_trans(features, transformations, max_angle, cutout_perc, noise_std, blur_std, i_iter)

        # Compute encodings using OMP
        X = omp(Y[indices_to_keep, :], D[indices_to_keep, :], n_nonzero_coefs)

        # Consider updating the atoms that are not used randomly
        row_sums = torch.sum(X != 0, dim=1)
        non_zero_row_indices = torch.nonzero(row_sums).squeeze()

        loss = simclr_loss(X)
        loss.backward()

        # Update dictionary using gradient descent through optimizer
        optimizer.step()

        # Optionally, normalize D after the gradient update, but ensure this does not affect gradients
        with torch.no_grad():
            D[:, non_zero_row_indices] = torch.randn(n_features, len(non_zero_row_indices), dtype=D.dtype)
            D[:] = F.normalize(D, dim=0)

    return D


def extract_data(feature_mat, label_mat, number):
    """
    Partition feature and label matrices into training and testing sets.
    
    Args:
    - featureMat (torch.Tensor): Feature matrix of shape (n_features, n_samples).
    - labelMat (torch.Tensor): Label matrix of shape (n_classes, n_samples), one-hot encoded.
    - number (int): Number of samples per class to include in the training set.
    
    Returns:
    - X_train (torch.Tensor): Training data features.
    - X_test (torch.Tensor): Testing data features.
    - y_train (torch.Tensor): Training data labels.
    - y_test (torch.Tensor): Testing data labels.
    """
    n_classes, n_samples = label_mat.shape
    n_features = feature_mat.shape[0]

    X_train = np.zeros((n_features, number * n_classes))
    X_test = np.zeros((n_features, n_samples - number * n_classes))
    y_train = np.zeros(number * n_classes)
    y_test = np.zeros(n_samples - number * n_classes)
    
    i_tr = 0
    i_tt = 0

    for i_class in range(n_classes):
        # Extract samples for the current class
        temp_data = feature_mat[:, label_mat[i_class, :] == 1]
        temp_label = torch.full((temp_data.shape[1],), i_class, dtype=torch.float)

        l = temp_label.shape[0]
        index = torch.randperm(l)
        
        X_train[:, i_tr:i_tr+number] = temp_data[:, index[:number]] if temp_data[:, index[:number]].ndim > 1 else temp_data[:, index[:number]].reshape(-1, 1)
        y_train[i_tr:i_tr+number] = temp_label[index[:number]]
        X_test[:, i_tt:i_tt+(l-number)] = temp_data[:, index[number:]] if temp_data[:, index[number:]].ndim > 1 else temp_data[:, index[number:]].reshape(-1, 1)
        y_test[i_tt:i_tt+(l-number)] = temp_label[index[number:]]

        i_tr += number
        i_tt += l - number
    
    return X_train, X_test, y_train, y_test
