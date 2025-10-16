import metrics
from irregular_grids import BoundedGrid

from typing import Callable # noqa: F401
import warnings # noqa: F401

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
# from scipy.spatial import Delaunay  # noqa: F401
from matplotlib import pyplot as plt

# warnings.filterwarnings("error")

class HeatGeodesicPaths:
    def __init__(self, rmetric:metrics.RMetric, grid:BoundedGrid, dim:int = 2, **kwargs):
        self.rmetric = rmetric
        self.grid = grid
        if dim != grid.dim:
            raise ValueError(f"Dimension mismatch: {dim} != {grid.dim}")
        if dim != rmetric.dim:
            raise ValueError(f"Dimension mismatch: {dim} != {rmetric.dim}")
        if dim != 2:
            raise NotImplementedError("Currently only 2D grids are supported.")
        self.dim = 2 # It is assumed that the grid is 2D for now.
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
            sqrtdetg = np.sqrt(self.rmetric.metric_det(x0))
            
            deltas = self.grid.deltas

            # make a 3*3 matrix of neighbors
            neighbors = np.array([[self.grid.neighbor(row, delta1 + delta2) for delta1 in (np.array((-1,0), dtype=int), np.array((0,0), dtype=int), np.array((1,0), dtype=int))]
                for delta2 in (np.array((0,-1), dtype=int), np.array((0,0), dtype=int), np.array((0,1), dtype=int))])

            if np.any(neighbors < 0):
                # If any neighbor is invalid, skip this row
                continue
            # legal_neighbors = neighbors.flatten() >= 0

            xm0 = (self.grid.idx_to_point(neighbors[0, 0]) + x0) / 2
            xp0 = (self.grid.idx_to_point(neighbors[2, 0]) + x0) / 2

            xm1 = (self.grid.idx_to_point(neighbors[0, 0]) + x0) / 2
            xp1 = (self.grid.idx_to_point(neighbors[0, 2]) + x0) / 2

            A_m0 = self.rmetric.metric_det(xm0) * self.rmetric.inv_metric(xm0)
            A_p0 = self.rmetric.metric_det(xp0) * self.rmetric.inv_metric(xp0)
            A_m1 = self.rmetric.metric_det(xm1) * self.rmetric.inv_metric(xm1)
            A_p1 = self.rmetric.metric_det(xp1) * self.rmetric.inv_metric(xp1)

            # Compute the weights for the Laplacian
            row_indices.extend([row,] *  9) #np.sum(legal_neighbors))
            col_indices.extend(neighbors.flatten()) #[legal_neighbors])
            W.extend(np.array([
                (   A_m0[0,1] + A_m1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg),
                ( - A_p0[0,1] + A_m0[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg) + A_m1[1,1] / (deltas[1]**2 * sqrtdetg),
                ( - A_p0[0,1] - A_m1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg),
                ( - A_p1[0,1] + A_m1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg) + A_m0[0,0] / (deltas[0]**2 * sqrtdetg),
                ( - A_p0[0,0] - A_m0[0,0]) / (deltas[0]**2 * sqrtdetg) + ( - A_p1[1,1] - A_m1[1,1]) / (deltas[1]**2 * sqrtdetg),
                (   A_p1[0,1] - A_m1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg) + A_p0[0,0] / (deltas[0]**2 * sqrtdetg),
                ( - A_m0[0,1] - A_p1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg),
                (   A_p0[0,1] - A_m0[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg) + A_p1[0,0] / (deltas[1]**2 * sqrtdetg),
                (   A_p0[0,1] + A_p1[0,1]) / (2 * deltas[0] * deltas[1] * sqrtdetg),
            ]))
            # ])[legal_neighbors])

            sum_spacing = sum_spacing + \
                np.sqrt(self.rmetric.metric(x0)[0, 0]) * self.grid.deltas[0] + \
                np.sqrt(self.rmetric.metric(x0)[1, 1]) * self.grid.deltas[1]
            spacing_counter += 2

        L = sparse.coo_matrix((W, (row_indices, col_indices)), shape=(self.grid.bounded_size, self.grid.bounded_size))
        L = sparse.csr_matrix(L)

        return L, sum_spacing / spacing_counter

    def heat_method(self, source, t_mult=.5):
        # Compute Laplacian
        L, h = self.L, self.h

        # Compute time step
        t = t_mult * (h ** 2)

        # Solve heat equation
        u0 = np.zeros(self.grid.bounded_size)
        u0[source] = 1.0
        u = spla.spsolve(sparse.eye(self.grid.bounded_size) - t * L, u0)
        # for _ in range(100):
        #     u = spla.spsolve(sparse.eye(self.grid.bounded_size) - t/100 * L, u)

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
                if i_p < 0 and i_m < 0:
                    raise ValueError(f"Both neighbors of row {row} in dimension {delta1dim} are invalid.")
                elif i_p < 0:
                    i_p = row
                    factor = 1
                elif i_m < 0:
                    i_m = row
                    factor = 1
                else:
                    factor = 2
                x_p = self.grid.idx_to_point(i_p)
                x_m = self.grid.idx_to_point(i_m)
                grad_contravariant[delta1dim] = (u[i_p] - u[i_m]) / factor / self.grid.deltas[delta1dim]
            inv_metric = self.rmetric.inv_metric(x0)
            grad_u[row] = np.einsum( "ij, j" , inv_metric, grad_contravariant)

        # Normalize the gradient
        X =grad_u / np.array([self.rmetric.geonorm(p, grad) if np.linalg.norm(grad) > 0 else 1.0 
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
                if i_p < 0 and i_m < 0:
                    raise ValueError(f"Both neighbors of row {row} in dimension {delta1dim} are invalid.")
                elif i_p < 0:
                    i_p = row
                    factor = 1
                elif i_m < 0:
                    i_m = row
                    factor = 1
                else:
                    factor = 2
                x_p = self.grid.idx_to_point(i_p)
                x_m = self.grid.idx_to_point(i_m)
                sqrtdetg = np.sqrt(self.rmetric.metric_det(x0))
                sqrtdetg_p = np.sqrt(self.rmetric.metric_det(x_p))
                sqrtdetg_m = np.sqrt(self.rmetric.metric_det(x_m))
                div[row] += (
                    (sqrtdetg_p * X[i_p, delta1dim]) - (sqrtdetg_m * X[i_m, delta1dim]) 
                ) / factor / sqrtdetg / self.grid.deltas[delta1dim]

        # Solve Poisson equation
        legalidxs = np.intersect1d(np.unique(L.nonzero()[0]), np.unique(L.nonzero()[1]))
        phi = np.full(self.grid.bounded_size, np.nan)
        phi[legalidxs] = spla.spsolve(L[legalidxs, :][:,legalidxs], div[legalidxs])

        # Shift minimum to zero
        phi -= np.nanmin(phi)

        return phi

# Example usage:
if __name__ == "__main__":
    from irregular_grids import BoundedGrid
    from metrics import AntiFerro

    # Create a dummy grid and metric
    aFmetric = AntiFerro()
    grid = BoundedGrid(cartesian_boundaries=[(0.1, 0.999), (-1.25, 1.25)], deltas=[0.02, 0.02], dim=2, bound_function = aFmetric.is_ordered_phase)

    # Initialize HeatGeodesicPaths
    heat_paths = HeatGeodesicPaths(aFmetric, grid)

    # Compute heat geodesic paths from a source point
    source_idx = grid.point_to_idx(np.array([0.2, 0.3]))
    phi = heat_paths.heat_method(source_idx)

    plt.scatter(grid.valid_points[:, 0], grid.valid_points[:, 1], s=0.5)
    plt.scatter(grid.valid_points[:, 0], grid.valid_points[:, 1], c=phi, cmap='viridis')
    plt.colorbar()
    plt.show()
    print("Computed heat geodesic lengths")
    print(phi)