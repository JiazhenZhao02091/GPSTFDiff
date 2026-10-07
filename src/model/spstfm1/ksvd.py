from src.model.spstfm1.omp import orthogonal_matching_pursuit_matrix
import torch
from torch import nn
import numpy as np


class KSVD(nn.Module):
    def __init__(
        self,
        atom_num,
        max_iter=10,
        tol=1e-6,
        sparsity=None,
        init_method='random',
        given_matrix=None,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.atom_num = atom_num
        self.sparsity = sparsity if sparsity is not None else int(atom_num * 0.1)
        self.init_method = init_method
        self.given_matrix = given_matrix

    def fit(self, observed_matrix):
        dictionary_matrix = self._initialize(observed_matrix)
        err_last = 0
        for i in range(self.max_iter):
            sparse_matrix = orthogonal_matching_pursuit_matrix(
                observed_matrix, dictionary_matrix, sparsity=self.sparsity
            )


            err_now = torch.mean(
                (observed_matrix - dictionary_matrix @ sparse_matrix) ** 2
            )


            print(f'Iteration {i}, error: {err_now:.6f}')
            if err_now < self.tol:
                break
            dictionary_matrix, sparse_matrix = self._update_dict(
                observed_matrix, dictionary_matrix, sparse_matrix
            )

        self.dictionary_matrix = dictionary_matrix
        self.sparse_matrix = sparse_matrix
        return dictionary_matrix, sparse_matrix

    def _initialize(self, observed_matrix):
        device = observed_matrix.device
        if self.init_method == 'random':
            dictionary_matrix = torch.randn(
                observed_matrix.shape[0], self.atom_num, device=device
            )
        elif self.init_method == 'data_elements':
            dictionary_matrix = observed_matrix[:, : self.atom_num]
        elif self.init_method == 'svd':
            pass
        elif self.init_method == 'given_matrix':
            dictionary_matrix = self.given_matrix
        dictionary_matrix = dictionary_matrix / torch.norm(
            dictionary_matrix, dim=0, keepdim=True
        )
        return dictionary_matrix

    def _update_dict(self, observed_matrix, dictionary_matrix, sparse_matrix):
        for j in range(self.atom_num):
            index = torch.nonzero(sparse_matrix[j, :])[:, 0]
            if len(index) == 0:
                continue
            dictionary_matrix[:, j] = 0
            residual = (observed_matrix - dictionary_matrix @ sparse_matrix)[:, index]
            u, sigma, v = torch.svd(residual)
            dictionary_matrix[:, j] = u[:, 0]
            sparse_matrix[j, index] = sigma[0] * v[:, 0]
        return dictionary_matrix, sparse_matrix


if __name__ == '__main__':
    import torch
    import os
    import numpy as np
    import random

    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    measurment_matrix = np.random.randn(200, 1000)
    ksvd = KSVD(atom_num=128, init_method='random')
