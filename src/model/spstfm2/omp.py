import torch


def orthogonal_matching_pursuit(
    observed_vector, dictionary_matrix, sparsity, err_treshold=1e-6
):


    _, atom_num = dictionary_matrix.shape
    subsapce_index_list = []
    residual = observed_vector
    k = 1
    sparse_vector = torch.zeros(atom_num, device=dictionary_matrix.device)
    err = torch.sum(residual**2)
    while k <= sparsity and err > err_treshold:
        residual_projection_vector = torch.abs(dictionary_matrix.T @ residual)
        _, matching_basis_index = torch.max(residual_projection_vector, dim=0)
        subsapce_index_list.append(matching_basis_index.item())
        subspace_matrix = dictionary_matrix[:, subsapce_index_list]
        subspace_matrix_pinv = torch.pinverse(subspace_matrix)
        subspace_coordinates = subspace_matrix_pinv @ observed_vector
        subspace_representation = subspace_matrix @ subspace_coordinates
        residual = observed_vector - subspace_representation
        err = torch.sum(residual**2)
        k += 1
    sparse_vector[subsapce_index_list] = subspace_coordinates
    return sparse_vector


def orthogonal_matching_pursuit_matrix(
    observed_matrix, dictionary_matrix, sparsity, err_treshold=1e-6
):
    _, sample_num = observed_matrix.shape
    _, atom_num = dictionary_matrix.shape
    sparse_matrix = torch.zeros(atom_num, sample_num, device=dictionary_matrix.device)
    for sample_idx in range(sample_num):
        observed_vector = observed_matrix[:, sample_idx]
        sparse_vector = orthogonal_matching_pursuit(
            observed_vector,
            dictionary_matrix,
            sparsity=sparsity,
            err_treshold=err_treshold,
        )
        sparse_matrix[:, sample_idx] = sparse_vector

    return sparse_matrix
