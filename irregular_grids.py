import numpy as np

class BoundedGrid(object):
    def __init__(self, dim, cartesian_boundaries, deltas, bound_function):
        """
        dim: int 
            The number of dimensions of the grid.
        cartesian_boundaries: list of tuples
            Each tuple contains the lower and upper bounds for each dimension.
        deltas: list of floats
            The step size for each dimension.
        bound_function: callable
            A function that takes a point and returns whether it is within the bounds.
        """
        assert dim == len(cartesian_boundaries) == len(deltas), "Dimension mismatch between dim, boundaries, and deltas."
        self.dim = dim
        self.cartesian_boundaries = np.array(cartesian_boundaries)
        self.checkbounds = bound_function
        self.deltas = np.array(deltas)
        
        xs = np.meshgrid(*[np.arange(cartesian_boundaries[i][0], cartesian_boundaries[i][1], deltas[i]) for i in range(dim)], indexing='ij')
        self.points = np.vstack([x.flatten() for x in xs]).T
        self.valid_idxs = np.array([i for i,p in enumerate(self.points) if bound_function(p)])
        self.bounded_size = len(self.valid_idxs)
        self.valid_points = self.points[self.valid_idxs]
        self.idxgrid = -np.ones(xs[0].shape, dtype=int)
        self.idxgrid[np.unravel_index(self.valid_idxs, xs[0].shape)] = np.arange(self.bounded_size)

        self.boundary = []
        for idx in range(self.bounded_size):
            for delta in np.eye(self.dim, dtype=int):
                if self.neighbor(idx, delta) == -1:
                    self.boundary.append(idx)
                    break
            for delta in -np.eye(self.dim, dtype=int):
                if self.neighbor(idx, delta) == -1:
                    self.boundary.append(idx)
                    break


    def point_to_idx(self, point):
        """
        Convert a point to the closest index in the grid.
        """
        idx = np.argmin(np.linalg.norm(self.valid_points - point, axis=1))
        return idx

    def idx_to_gridpoint(self, idx):
        """
        Convert an index to a grid point.
        """
        if idx<0 or idx>=self.bounded_size:
            return -1
        return np.array(np.where(idx == self.idxgrid)).flatten()
    
    def idx_to_point(self, idx):
        """
        Convert an index to a point in the grid.
        """
        if idx<0 or idx>=self.bounded_size:
            return -1
        return self.valid_points[idx]

    def neighbor(self, idx, grid_delta):
        """
        Get the neighbor index of a point given a delta in each dimension.
        """
        grid_point = self.idx_to_gridpoint(idx)
        new_gridpoint = grid_point + grid_delta
        if np.any(new_gridpoint < 0) or np.any(new_gridpoint >= self.idxgrid.shape):
            return -1
        new_idx = self.idxgrid[*new_gridpoint]
        return new_idx
    
    def values_to_grid(self, values):
        """
        Convert a list of values of length self.bounded_size to a grid representation.
        """
        grid = np.full(self.idxgrid.shape, np.nan)
        for idx, value in zip(self.valid_idxs, values):
            grid[np.unravel_index(idx, self.idxgrid.shape)] = value
        return grid