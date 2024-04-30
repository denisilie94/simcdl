import torch
import torch.nn.functional as F


def omp(Y, D, n_nonzero_coefs):
    """
    Orthogonal Matching Pursuit (OMP) implementation in PyTorch.

    Args:
    - Y (Tensor): Matrix of signals of size (m, N).
    - D (Tensor): Dictionary matrix of size (m, n), where m is the number of features and n is the number of atoms.
    - n_nonzero_coefs (int): Desired sparsity level (number of non-zero entries in the solution).

    Returns:
    - X (Tensor): Sparse solution matrix of size (n, N).
    """
    Dt = D.t()

    n_samples = Y.shape[1]
    n_atoms = D.shape[1]

    X = torch.zeros(n_atoms, n_samples, device=D.device, dtype=D.dtype)

    for idx in range(n_samples):
        y = Y[:, idx:idx+1]
        residual = y.clone()
        index_set = []
    
        for _ in range(n_nonzero_coefs):
            correlations = Dt.mm(residual)
            _, max_idx = correlations.abs().max(0)
            max_idx = max_idx.item()
    
            if max_idx not in index_set:
                index_set.append(max_idx)
    
            D_selected = D[:, index_set]
            DtD = D_selected.t().mm(D_selected)
            DtY = D_selected.t().mm(y)
            x_ls = torch.linalg.solve(DtD, DtY)

            # Sparse representation update
            X[index_set, idx] = x_ls.squeeze()

            # Efficient residual update
            residual = y - D_selected.mm(x_ls)

    return X


def normc(tensor):
    """Normalize the columns of a 2D tensor"""
    return F.normalize(tensor, p=2, dim=0)


def ak_svd_update(Y, D, n_nonzero_coefs):
    """
    Approximate K-SVD update function
    Args:
        Y: Input matrix
        D: Dictionary matrix
        n_nonzero_coefs: Number of nonzero coefficients for OMP
    Returns:
        D: Updated dictionary matrix
        X: Sparse representation matrix
    """
    X = omp(Y, D, n_nonzero_coefs)
    E = Y - D @ X

    for i_atom in range(D.size(1)):
        atom_usages = X[i_atom, :].nonzero().squeeze(1)

        if atom_usages.nelement() == 0:
            D[:, i_atom:i_atom+1] = normc(torch.randn((D.size(0), 1), dtype=D.dtype))
        else:
            if atom_usages.dim() == 0:
                atom_usages = atom_usages.unsqueeze(0)

            F = E[:, atom_usages] + D[:, i_atom:i_atom+1] @ X[i_atom:i_atom+1, atom_usages]
            d = F @ X[i_atom, atom_usages].t()
            D[:, i_atom] = normc(d).squeeze()
            X[i_atom, atom_usages] = (F.t() @ D[:, i_atom:i_atom+1]).squeeze()
            E[:, atom_usages] = F - D[:, i_atom:i_atom+1] @ X[i_atom:i_atom+1, atom_usages]

    return D, X


def red_ak_svd_update(Y, D, n_nonzero_coefs, rows):
    """
    Reduced Approximate K-SVD update function
    Args:
        Y: Input matrix
        D: Dictionary matrix
        n_nonzero_coefs: Number of nonzero coefficients for OMP
        rows: rows index that are going to be used during optimization
    Returns:
        D: Updated dictionary matrix
        X: Sparse representation matrix
    """
    X = omp(Y, D, n_nonzero_coefs)
    E = Y[rows, :] - D[rows, :] @ X

    for i_atom in range(D.shape[1]):
        atom_usages = X[i_atom, :].nonzero().squeeze(1)

        if len(atom_usages) == 0:
            D[rows, i_atom:i_atom+1] = torch.randn((len(rows), 1), dtype=D.dtype)
            D[:, i_atom] = normc(D[:, i_atom])
        else:
            if atom_usages.dim() == 0:
                atom_usages = atom_usages.unsqueeze(0)

            F = E[:, atom_usages] + D[rows, i_atom:i_atom+1] @ X[i_atom:i_atom+1, atom_usages]
            d = F @ X[i_atom, atom_usages].t()
            D[rows, i_atom] = d
            D[:, i_atom] = normc(D[:, i_atom])
            X[i_atom, atom_usages] = (F.t() @ D[rows, i_atom:i_atom+1]).squeeze()
            E[:, atom_usages] = F - D[rows, i_atom:i_atom+1] @ X[i_atom:i_atom+1, atom_usages]

    return D, X
