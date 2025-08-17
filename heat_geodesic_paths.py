import metrics
from irregular_grids import BoundedGrid

from typing import Callable # noqa: F401
import warnings # noqa: F401

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
# from scipy.spatial import Delaunay  # noqa: F401
from matplotlib import pyplot as plt

class HeatGeodesicPaths:
    def __init__(self, rmetric:metrics.RMetric, grid:BoundedGrid, dim: int = 2, **kwargs):
        self.rmetric = rmetric
        self.grid = grid
        self.dim = dim
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.L, self.h = self.laplacian()

    def laplacian(self):
        row_indices = []
        col_indices = []
        W = []
        sum_spacing = 0
        spacing_counter = 0
        for row in np.arange(self.grid.bounded_size):
            x0 = self.grid.idx_to_point(row)
            for delta1dim in np.arange(self.grid.dim):
                delta1 = np.zeros(self.grid.dim, dtype=int)
                delta1[delta1dim] = 1
                i1p = self.grid.neighbor(row, delta1)
                i1m = self.grid.neighbor(row, -delta1)
                if i1p >= 0 and i1m >= 0:
                    x1p = self.grid.idx_to_point(i1p)
                    x1m = self.grid.idx_to_point(i1m)
                    dx1 = delta1 * self.grid.deltas**-1
                    g_inv_1p = self.rmetric.inv_metric(x1p)
                    g_inv_1m = self.rmetric.inv_metric(x1m)
                    for delta2dim in np.arange(self.dim):
                        delta2 = np.zeros(self.dim, dtype=int)
                        delta2[delta2dim] = 1
                        i2pp = self.grid.neighbor(i1p, delta2)
                        i2pm = self.grid.neighbor(i1p, -delta2)
                        i2mp = self.grid.neighbor(i1m, delta2)
                        i2mm = self.grid.neighbor(i1m, -delta2)
                        if np.all(np.array([i2pp, i2pm, i2mp, i2mm]) >= 0):
                            dx2 = delta2 * self.grid.deltas
                            detg = self.rmetric.metric_det(x0)
                            detg_1p = self.rmetric.metric_det(x1p)
                            detg_1m = self.rmetric.metric_det(x1m)
                            row_indices.extend([row] * 4)
                            col_indices.extend([i2pp, i2pm, i2mp, i2mm])
                            W.extend([
                                (detg_1p * np.dot(dx1, np.dot(g_inv_1p, dx2))) / 4 / detg,
                                -(detg_1p * np.dot(dx1, np.dot(g_inv_1p, dx2))) / 4 / detg,
                                -(detg_1m * np.dot(dx1, np.dot(g_inv_1m, dx2))) / 4 / detg,
                                (detg_1m * np.dot(dx1, np.dot(g_inv_1m, dx2))) / 4 / detg
                            ])
                            sum_spacing = sum_spacing + \
                                np.einsum("ij, i, j", self.rmetric.metric(x0), dx1, dx1) + \
                                np.einsum("ij, i, j", self.rmetric.metric(x0), dx2, dx2)
                        spacing_counter += 2

        L = sparse.coo_matrix((W, (row_indices, col_indices)), shape=(self.grid.bounded_size, self.grid.bounded_size))
        L = sparse.csr_matrix(L)

        return L, sum_spacing / spacing_counter

    def heat_method(self, source, t_mult=1.0):
        # Compute Laplacian
        L, h = self.L, self.h

        print(L)
        # Compute time step
        t = t_mult * (h ** 2)

        # Solve heat equation
        u0 = np.zeros(self.grid.bounded_size)
        u0[source] = 1.0
        u = spla.spsolve(sparse.eye(self.grid.bounded_size) + t * L, u0)

        # Compute gradient of u
        grad_u = np.zeros((self.grid.bounded_size, self.grid.dim))
        for row in np.arange(self.grid.bounded_size):
            x0 = self.grid.idx_to_point(row)
            grad_contravariant = np.zeros(self.grid.dim)
            for delta1dim in np.arange(self.grid.dim):
                delta1 = np.zeros(self.grid.dim, dtype=int)
                delta1[delta1dim] = 1
                i_p = self.grid.neighbor(row, delta1)
                i_m = self.grid.neighbor(row, -delta1)
                if i_p >= 0 and i_m >= 0:
                    x_p = self.grid.idx_to_point(i_p)
                    x_m = self.grid.idx_to_point(i_m)
                    inv_metric = self.rmetric.inv_metric(x0)
                    grad_contravariant[delta1dim] = (u[i_p] - u[i_m]) / 2 / self.grid.deltas[delta1dim]
                    grad_u[row] = np.einsum( "ij, j" , inv_metric, grad_contravariant)

        # Normalize the gradient
        X = grad_u / np.array([self.rmetric.geonorm(p, grad) 
            for p, grad in zip(self.grid.valid_points, grad_u)])[:, np.newaxis]
        
        # Compute divergence
        div = np.zeros(self.grid.bounded_size)
        for row in np.arange(self.grid.bounded_size):
            x0 = self.grid.idx_to_point(row)
            for delta1dim in np.arange(self.grid.dim):
                delta1 = np.zeros(self.grid.dim, dtype=int)
                delta1[delta1dim] = 1
                i_p = self.grid.neighbor(row, delta1)
                i_m = self.grid.neighbor(row, -delta1)
                if i_p >= 0 and i_m >= 0:
                    x_p = self.grid.idx_to_point(i_p)
                    x_m = self.grid.idx_to_point(i_m)
                    detg = self.rmetric.metric_det(x0)
                    detg_p = self.rmetric.metric_det(x_p)
                    detg_m = self.rmetric.metric_det(x_m)
                    div[row] += (
                        (detg_p * X[i_p, delta1dim]) / 2 / detg / self.grid.deltas[delta1dim] -
                        (detg_m * X[i_m, delta1dim]) / 2 / detg / self.grid.deltas[delta1dim]
                    )

        # Solve Poisson equation
        phi = spla.spsolve(L, div)

        # Shift minimum to zero
        phi -= np.min(phi)

        return phi

# Example usage:
if __name__ == "__main__":
    from irregular_grids import BoundedGrid
    from metrics import AntiFerro

    # Create a dummy grid and metric
    aFmetric = AntiFerro()
    grid = BoundedGrid(cartesian_boundaries=[(0.001, 0.999), (-1.225, 1.25)], deltas=[0.1, 0.1], dim=2, bound_function = aFmetric.is_ordered_phase)

    # Initialize HeatGeodesicPaths
    heat_paths = HeatGeodesicPaths(aFmetric, grid)

    # Compute heat geodesic paths from a source point
    source_idx = grid.point_to_idx(np.array([0.2, 0.3]))
    phi = heat_paths.heat_method(source_idx)

    print("Computed heat geodesic lengths")
    print(phi)
    plt.scatter(grid.valid_points[:, 0], grid.valid_points[:, 1], c=phi, cmap='viridis')
    plt.show()