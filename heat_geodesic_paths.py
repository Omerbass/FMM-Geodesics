from typing import Callable
import warnings # noqa: F401

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay  # noqa: F401


class HeatGeodesicPaths:
    def __init__(self, metric: Callable[[np.ndarray], np.ndarray], dim: int = 2, **kwargs):
        self.metric = metric
        self.dim = dim
        for key, val in kwargs.items():
            setattr(self, key, val)

    def laplacian(self, V, F):
        # V: vertices, F: faces
        i1, i2, i3 = F[:,0], F[:,1], F[:,2]
        v1, v2, v3 = V[i1], V[i2], V[i3]
        m1 = [self.metric(v) for v in v1]
        m2 = [self.metric(v) for v in v2]
        m3 = [self.metric(v) for v in v3]
        e1, e2, e3 = v2 - v3, v3 - v1, v1 - v2

        cos1 = np.einsum('ij, ijk, ik->i', e2, m1, e3) / np.sqrt(np.einsum('ij, ijk, ik->i', e2, m1, e2) * np.einsum('ij, ijk, ik->i', e3, m1, e3))
        cos2 = np.einsum('ij, ijk, ik->i', e3, m2, e1) / np.sqrt(np.einsum('ij, ijk, ik->i', e3, m2, e3) * np.einsum('ij, ijk, ik->i', e1, m2, e1))
        cos3 = np.einsum('ij, ijk, ik->i', e1, m3, e2) / np.sqrt(np.einsum('ij, ijk, ik->i', e1, m3, e1) * np.einsum('ij, ijk, ik->i', e2, m3, e2))

        cot1 = - cos1 / np.sqrt(1 - cos1**2)
        cot2 = - cos2 / np.sqrt(1 - cos2**2)
        cot3 = - cos3 / np.sqrt(1 - cos3**2)

        row_indices = np.concatenate([i2, i3, i3, i1, i1, i2])
        col_indices = np.concatenate([i3, i2, i1, i3, i2, i1])
        W = 0.5 * np.concatenate([cot1, cot1, cot2, cot2, cot3, cot3])

        L = sparse.coo_matrix((W, (row_indices, col_indices)), shape=(V.shape[0], V.shape[0]))
        L = sparse.csr_matrix(L)
        L = sparse.diags(L.sum(axis=1).A1) - L

        return L

    def heat_method(self, V, F, source, t_mult=1.0):
        # Compute Laplacian
        L = self.laplacian(V, F)

        # Compute mass matrix (lumped)
        A = np.zeros(V.shape[0])
        for tri in F:
            area = np.linalg.norm(np.cross(V[tri[1]]-V[tri[0]], V[tri[2]]-V[tri[0]])) / 2
            A[tri] += area / 3

        M = sparse.diags(A)

        # Compute time step
        h = np.mean(np.sqrt(A))
        t = t_mult * (h ** 2)

        # Solve heat equation
        u0 = np.zeros(V.shape[0])
        u0[source] = 1.0
        u = spla.spsolve(M + t * L, M @ u0)

        # Compute gradient of u
        grad_u = np.zeros((F.shape[0], 3))
        for i, tri in enumerate(F):
            v0, v1, v2 = V[tri]
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
            N = np.cross(v1 - v0, v2 - v0) / (2 * area)
            grad_u[i] = np.cross(N, (u[tri[1]] - u[tri[0]]) * (v2 - v0) + (u[tri[2]] - u[tri[0]]) * (v0 - v1)) / (2 * area)

        # Normalize the gradient
        X = grad_u / np.linalg.norm(grad_u, axis=1)[:, np.newaxis]

        # Compute divergence
        div = np.zeros(V.shape[0])
        for i, tri in enumerate(F):
            for j in range(3):
                j1, j2 = tri[j], tri[(j+1)%3]
                edge = V[j2] - V[j1]
                weight = np.dot(X[i], edge)
                div[j1] += weight / 2
                div[j2] += weight / 2

        # Solve Poisson equation
        phi = spla.spsolve(L, div)

        # Shift minimum to zero
        phi -= np.min(phi)

        return phi

# Example usage:
# vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
# faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
# source_vertex = 0
# distance = heat_method(vertices, faces, source_vertex)
