import numpy as np

def structured_grid_triangles(nx, ny):
    """
    Triangulate a plain rectangular nx*ny grid into 2 triangles per cell,
    without scipy.spatial.Delaunay/Qhull. Positions must be laid out as
    np.reshape(np.meshgrid(x, y), (2, -1)).T for x of length nx, y of
    length ny (index k = i*nx+j, i indexing y, j indexing x -- numpy's
    default meshgrid 'xy' indexing) -- i.e. exactly the pattern used by
    main_flat/main_sphere/main_spherical in tests/fmmGeodesics.py.

    Qhull's Delaunay is solving a much harder problem than this needs: a
    general point-position triangulation, including breaking ties among
    the many exactly-cocircular quadruples a perfect grid produces. Since
    the combinatorial structure of a rectangular grid is already known
    exactly, this builds the same triangle count directly, ~150x faster
    at 10^6 points (see git history). The diagonal alternates checkerboard-
    style per cell (rather than a single fixed direction) so the
    triangulation doesn't impose a systematic directional bias on
    Euclidean-coordinate quantities (e.g. FMM's per-triangle updates) --
    matching how Qhull tends to split a near-perfect grid anyway, without
    its general-position machinery.

    Only valid for a *complete* rectangular grid with no missing points
    (e.g. not a BoundedGrid with an irregular bound_function cutting holes
    out of it) -- use scipy.spatial.Delaunay for those.
    """
    i, j = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing='ij')
    i = i.ravel(); j = j.ravel()
    a = i * nx + j       # bottom-left
    b = a + 1             # bottom-right (x+1)
    c = a + nx            # top-left (y+1)
    d = c + 1              # top-right
    even = (i + j) % 2 == 0
    tri1 = np.where(even[:, None], np.stack([a, b, d], axis=1), np.stack([a, b, c], axis=1))
    tri2 = np.where(even[:, None], np.stack([a, d, c], axis=1), np.stack([b, d, c], axis=1))
    return np.concatenate([tri1, tri2], axis=0)

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