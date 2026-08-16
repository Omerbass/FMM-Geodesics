import numpy as np
import heapq
from rich.progress import Progress
import rich.progress as rprog
from rich import print as rprint # noqa: F401
import itertools as it
import warnings

# from . import metrics
# from .irregular_grids import BoundedGrid

warnings.filterwarnings("error", category=RuntimeWarning)

class FMMGeodesicPaths:
    def __init__(self, metric, dim=2, inv_metric=None, **kwargs):
        if inv_metric is None:
            self.inv_metric = lambda x: np.linalg.inv(metric(x))
        else:
            self.inv_metric = inv_metric
        self.metric = metric
        self.dim = dim
        for key, val in kwargs.items():
            setattr(self, key, val)

    def dist(self, a, b):
        """
        Compute the infenitesimal distance between two points.
        """
        diff = b - a
        return np.sqrt(diff.T @ self.metric((a+b)/2) @ diff)
    
    def geonorm(self, p, a):
        """
        Compute the norm of a vector, on the geodesic at point p.
        """
        sqrnorm = a.T @ self.metric(p) @ a
        if np.isclose(sqrnorm, 0, atol=1e-12, rtol=0) and sqrnorm < 0:
            return 1e-25
        elif sqrnorm < 0:
            raise RuntimeWarning(f"Negative squared norm {sqrnorm} at point {p} for vector {a}.\nMetric:\n{self.metric(p)}")
        return np.sqrt(sqrnorm)
    
    def geoip(self, p, a, b):
        """
        Compute the inner product of two vectors, on the geodesic at point p.
        """
        return a.T @ self.metric(p) @ b

    def solve_quadratic(self, a, b, c):
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return np.inf
        return (-b + np.sqrt(disc)) / (2 * a)

    def is_obtuse_triangle(self, a, b, c):
        edges = [self.dist(a, b), self.dist(b, c), self.dist(c, a)]
        edges.sort()
        return edges[2]**2 > edges[0]**2 + edges[1]**2

    def _safe_norm(self, sqrnorm, p, vec):
        """Same negative-squared-norm handling as geonorm(), factored out so
        update_triangle can reuse a single metric evaluation instead of
        calling geonorm/geoip (each of which evaluates the metric again)."""
        if sqrnorm < 0:
            if np.isclose(sqrnorm, 0, atol=1e-12, rtol=0):
                return 1e-25
            raise RuntimeWarning(f"Negative squared norm {sqrnorm} at point {p} for vector {vec}.\nMetric:\n{self.metric(p)}")
        return np.sqrt(sqrnorm)

    def update_triangle(self, T, positions, triangles, A, B, C, F=1.0):
        Ta, Tb, Tc = T[A], T[B], T[C]

        # Ensure Ta <= Tb
        if Ta > Tb:
            Ta, Tb = Tb, Ta
            A, B = B, A

        u = Tb- Ta

        if np.isinf(u):
            return np.inf

        vec_ab = positions[B] - positions[A]
        vec_ac = positions[C] - positions[A]

        # geonorm(A,vec_ab), geonorm(A,vec_ac) and geoip(A,vec_ab,vec_ac) each
        # independently evaluated self.metric(positions[A]) before -- three
        # calls to the same (potentially expensive) metric at the same point.
        # One evaluation, reused, is equivalent and much cheaper.
        gA = self.metric(positions[A])
        a = self._safe_norm(vec_ab @ gA @ vec_ab, positions[A], vec_ab)
        b = self._safe_norm(vec_ac @ gA @ vec_ac, positions[A], vec_ac)
        cos_theta = (vec_ab @ gA @ vec_ac) / (a * b)
        if np.abs(cos_theta) > 1 and np.isclose(cos_theta**2, 1, rtol=0):
            cos_theta = np.sign(cos_theta)
        sin_theta = np.sqrt(1 - cos_theta**2)

        if a**2 + b**2 - 2*a*b*cos_theta == 0:
            t = np.inf
        else:
            t = self.solve_quadratic(a**2 + b**2 - 2*a*b*cos_theta,
                                2*b*u*(a*cos_theta - b),
                                b**2*(u**2 - (F**2)*(a**2)*(sin_theta**2)))
        
        try:
            if u < t and (a * cos_theta <  b * (1 - u/t)) and\
                    (b * (1 - u/t) < (a / cos_theta if cos_theta != 0 else np.inf)):
                Tc_new = min(Tc, Ta + t)
            else:
                Tc_new = min(Tc, Ta + b * F, Tb + self.dist(positions[C], positions[B]) * F)
        except RuntimeWarning:
            print(t, cos_theta, sin_theta, a, b, u)
            Tc_new = Ta + t


        return min(Tc, Tc_new)

    def fast_marching_method(self, positions, triangles, source):
        num_points = positions.shape[0]
        triangles = np.asarray(triangles)

        # point -> list of incident triangle row-indices, built once. Previously
        # every popped point rescanned the *entire* triangle list
        # ([tri for tri in triangles if p in tri]), making the whole method
        # O(num_points * num_triangles); this makes each lookup O(degree).
        point_tris = [[] for _ in range(num_points)]
        for ti, tri in enumerate(triangles):
            for v in tri:
                point_tris[v].append(ti)

        FAR, CLOSE, ALIVE = 0, 1, 2
        status = np.full(num_points, FAR, dtype=np.int8)
        T = np.full(num_points, np.inf)

        T[source] = 0.0
        status[source] = ALIVE
        n_alive, n_close = 1, 0

        heap = []
        for ti in point_tris[source]:
            for p in triangles[ti]:
                if p != source and status[p] == FAR:
                    status[p] = CLOSE
                    n_close += 1
                    T[p] = self.dist(positions[p], positions[source])
                    heapq.heappush(heap, (T[p], p))

        # Updating the rich progress bar every iteration serializes on its
        # live-refresh thread's lock and dominates runtime at any real N
        # (profiled: >2/3 of total time in thread-lock acquisition on a
        # 625-point grid); refresh only ~200 times over the whole run instead.
        progress_every = max(1, num_points // 200)

        with Progress(*Progress.get_default_columns(), rprog.TimeElapsedColumn(), rprog.MofNCompleteColumn()) as progress:
            task0 = progress.add_task("[cyan]heap", total=num_points)
            task1 = progress.add_task("[white]alive", total=num_points)
            task2 = progress.add_task("[yellow]close", total=num_points)
            n_popped = 0
            while heap:
                _, p = heapq.heappop(heap)
                if status[p] == ALIVE:
                    continue  # stale duplicate heap entry; T[p] already final
                status[p] = ALIVE
                n_alive += 1
                n_close -= 1

                n_popped += 1
                if n_popped % progress_every == 0:
                    progress.update(task0, completed=len(heap))
                    progress.update(task1, completed=n_alive)
                    progress.update(task2, completed=n_close)

                for ti in point_tris[p]:
                    tri = triangles[ti]
                    for q in tri:
                        if status[q] != ALIVE:
                            r = [v for v in tri if v != p and v != q][0]
                            if status[r] == ALIVE:
                                if np.isinf(T[p]) and np.isinf(T[r]):
                                    continue
                                T_old = T[q]
                                T[q] = self.update_triangle(T, positions, triangles, p, r, q)

                                if status[q] == FAR:
                                    heapq.heappush(heap, (T[q], q))
                                    status[q] = CLOSE
                                    n_close += 1
                                elif T[q] < T_old:
                                    heapq.heappush(heap, (T[q], q))
            progress.update(task0, completed=0)
            progress.update(task1, completed=n_alive)
            progress.update(task2, completed=n_close)

        return T

    # def gradient_descent_to_origin_along_simplices(self, grid, delaunay, distances, start, max_steps=1000, tol=1e-6):
    #     path = [grid.valid_points[start]]
    #     current = grid.valid_points[start]
    #     final_idx = np.argmin(distances)
    #     final_point = grid.valid_points[final_idx]

    #     def get_normalized_gradient(point, simplex):
    #         if simplex == -1:
    #             print(point)
    #             print("Warning: point outside of triangulation.")
    #             return np.zeros(self.dim)
    #         vertices = delaunay.simplices[simplex]
    #         d1, d2, d3 = distances[vertices]
    #         p1, p2, p3 = grid.valid_points[vertices]
    #         p_mat = np.column_stack((p2 - p1, p3 - p1))
    #         d_vec = np.array([d2 - d1, d3 - d1])
    #         # grad_embedded = p_mat @ np.linalg.solve(p_mat.T @ p_mat, d_vec)
    #         # grad = np.einsum("ij, j", self.inv_metric(point), grad_embedded)
    #         # grad_norm = self.geonorm(point, grad)
    #         grad = p_mat @ np.linalg.solve(p_mat.T @ p_mat, d_vec)
    #         grad_norm = np.linalg.norm(grad)
    #         return grad / grad_norm

    #     def intersect_lines(p1, p2, v1, v2):
    #         A = np.array([v1, -v2]).T
    #         if np.abs(np.linalg.det(A)) < 1e-12:
    #             return None
    #         b = p2 - p1
    #         t = np.linalg.solve(A, b)
    #         return (p1 + t[0] * v1, t[0], t[1])

    #     current_simplex = delaunay.find_simplex(current)

    #     for _ in range(max_steps):
    #         if final_idx in delaunay.simplices[current_simplex]:
    #             path.append(final_point)
    #             break
    #         elif self.dist(current, final_point) < tol:
    #             break
    #         grad = get_normalized_gradient(current, current_simplex)
    #         if not np.any(grad):
    #             current = grid.valid_points[grid.point_to_idx(current)]
    #             path.append(current)
    #             rprint(f"[red]Warning: {current} outside of grid.")
    #             return np.array(path)
    #         print("----", current, "----")
    #         print("triangle :", delaunay.simplices[current_simplex])
    #         print("gradient :", grad)
    #         for i1, i2 in it.combinations([0, 1,2], 2):
    #             ix1, ix2 = delaunay.simplices[current_simplex][[i1, i2]]
    #             p1, p2 = grid.valid_points[[ix1, ix2]]
    #             edge_vec = p2 - p1
    #             res = intersect_lines(current, p1, -grad, edge_vec)
    #             print("intersection :", res, " with edge ", p1, p2)
    #             if res is None:
    #                 continue
    #             intersection, t1, t2 = res
    #             if t1 > 1e-12 and 0 <= t2 <= 1:
    #                 current = intersection
    #                 path.append(current)
    #                 print(delaunay.neighbors[current_simplex], i1, i2)
    #                 current_simplex = delaunay.neighbors[current_simplex][list({0,1,2} - {i1, i2})[0]]
    #                 print("next simplex :", current_simplex, ", edges:", delaunay.simplices[current_simplex])
    #                 break
    #         else:
    #             rprint(f"[red]Warning: no intersection found for point {current}, stopping.")
    #             break

    #     return np.array(path)

    def gradient_descent_to_origin(self, grid, delaunay, distances, start, step_size=0.01, max_steps=1000, tol=1e-6):
        path = [grid.valid_points[start]]
        current = grid.valid_points[start]
        final_idx = np.argmin(distances)
        final_point = grid.valid_points[final_idx]

        # print("no inverse metric correction")

        # mesh boundary graph: vertex -> adjacent vertices sharing an edge with no
        # neighboring triangle (delaunay.neighbors == -1), i.e. the triangulation's edge
        boundary_neighbors = {}
        for tri, nbrs in zip(delaunay.simplices, delaunay.neighbors):
            for i in range(3):
                if nbrs[i] == -1:
                    a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
                    boundary_neighbors.setdefault(a, set()).add(b)
                    boundary_neighbors.setdefault(b, set()).add(a)
        boundary_vertex_ids = np.array(list(boundary_neighbors.keys()))
        boundary_vertex_pts = grid.valid_points[boundary_vertex_ids]

        def nearest_boundary_vertex(point):
            """Snap a point that's about to leave the mesh to the closest boundary vertex."""
            grid_idx = grid.point_to_idx(point)
            if grid_idx in boundary_neighbors:
                return grid_idx
            d2 = np.sum((boundary_vertex_pts - point) ** 2, axis=1)
            return boundary_vertex_ids[np.argmin(d2)]

        def would_exit(point, direction, step_len):
            """Would stepping step_len along direction from point leave the mesh?"""
            return delaunay.find_simplex(point + step_len * direction) == -1

        def discrete_grid_hop(point):
            """Fall back to the grid neighbor with the lowest distance (same greedy
            hop as gradient_descent_to_origin_along_grid), for use when the
            continuous RK2 step becomes unreliable. Returns None if no improving
            neighbor exists."""
            current_idx = grid.point_to_idx(point)
            neighbors = []
            for delta in it.product(*[[1, 0, -1], ] * grid.dim):
                if not np.any(delta):
                    continue
                neighbor_idx = grid.neighbor(current_idx, delta)
                if neighbor_idx != -1:
                    neighbors.append(neighbor_idx)
            if not neighbors:
                return None
            next_idx = min(neighbors, key=lambda idx: distances[idx])
            if distances[next_idx] > distances[current_idx] + tol:
                return None
            return next_idx

        def get_normalized_gradient(point):
            simplex = delaunay.find_simplex(point)
            if simplex == -1:
                print(point)
                print("Warning: point outside of triangulation.")
                return np.zeros(self.dim)
            
            vertices = delaunay.simplices[simplex]
            d1, d2, d3 = distances[vertices]
            p1, p2, p3 = grid.valid_points[vertices]
            p_mat = np.column_stack((p2 - p1, p3 - p1))
            d_vec = np.array([d2 - d1, d3 - d1])
            grad_embedded = p_mat @ np.linalg.solve(p_mat.T @ p_mat, d_vec)
            
            # Include gradients from neighboring triangles
            neighbors = delaunay.neighbors[simplex]
            for neighbor in neighbors:
                if neighbor != -1:
                    n_vertices = delaunay.simplices[neighbor]
                    nd1, nd2, nd3 = distances[n_vertices]
                    np1, np2, np3 = grid.valid_points[n_vertices]
                    n_p_mat = np.column_stack((np2 - np1, np3 - np1))
                    n_d_vec = np.array([nd2 - nd1, nd3 - nd1])
                    n_grad_embedded = n_p_mat @ np.linalg.solve(n_p_mat.T @ n_p_mat, n_d_vec)
                    grad_embedded += n_grad_embedded
            
            grad_embedded /= (1 + len([n for n in neighbors if n != -1]))  # Average over all triangles
            return grad_embedded / np.linalg.norm(grad_embedded)
            # grad = np.einsum("ij, j", self.inv_metric(point), grad_embedded)
            # grad_norm = self.geonorm(point, grad)
            # return grad / grad_norm

        boundary_vertex = None  # set to a vertex index while following the mesh boundary (1D)

        with Progress() as progress:
            task = progress.add_task("[cyan]distance", total=distances[start])
            for _ in range(max_steps):
                if boundary_vertex is not None:
                    # follow the mesh boundary downhill instead of stepping off the mesh
                    neighbors = boundary_neighbors.get(boundary_vertex, ())
                    next_idx = min(neighbors, key=lambda idx: distances[idx], default=None)
                    if next_idx is None or distances[next_idx] > distances[boundary_vertex] + tol:
                        break  # local minimum along the boundary; nothing more to do
                    boundary_vertex = next_idx
                    current = grid.valid_points[boundary_vertex]
                    path.append(current)
                    if self.dist(current, final_point) < tol:
                        break
                    progress.update(task, completed=distances[start] - self.dist(current, final_point))

                    inner_grad = get_normalized_gradient(current)
                    if np.any(inner_grad) and not would_exit(current, -inner_grad, step_size):
                        boundary_vertex = None  # gradient points back inside; resume normal descent
                    continue

                true_step = np.min((step_size, self.dist(current, final_point)))
                grad = get_normalized_gradient(current)
                if not np.any(grad):
                    current = grid.valid_points[grid.point_to_idx(current)]
                    path.append(current)
                    rprint(f"[red]Warning: {current} outside of grid.")
                    break

                if would_exit(current, -grad, true_step):
                    boundary_vertex = nearest_boundary_vertex(current)
                    current = grid.valid_points[boundary_vertex]
                    path.append(current)
                    if self.dist(current, final_point) < tol:
                        break
                    progress.update(task, completed=distances[start] - self.dist(current, final_point))
                    continue

                # RK2, with adaptive step-halving when predictor/corrector disagree:
                # if grad @ grad_pred < 0 the two point more than 90 degrees apart, so
                # their average collapses toward zero and the step becomes unreliable.
                rk2_step = true_step
                for _ in range(5):
                    next_pred = current - rk2_step * grad
                    grad_pred = get_normalized_gradient(next_pred)
                    if grad @ grad_pred >= 0:
                        break
                    rk2_step /= 2
                else:
                    # still disagreeing at the smallest step: the continuous estimate
                    # is unreliable here, so fall back to a discrete hop to the best
                    # neighboring grid vertex instead
                    next_idx = discrete_grid_hop(current)
                    if next_idx is not None:
                        current = grid.valid_points[next_idx]
                        path.append(current)
                        if self.dist(current, final_point) < tol:
                            break
                        progress.update(task, completed=distances[start] - self.dist(current, final_point))
                        continue

                nxt = current - 0.5 * rk2_step * (grad + grad_pred)

                current = nxt
                path.append(current)
                if self.dist(current, final_point) < tol:
                    break
                progress.update(task, completed=distances[start] - self.dist(current, final_point))
        return np.array(path)

    def gradient_descent_to_origin_along_grid(self, grid, distances, start, tol=1e-6):
        """ Gradient descent to the origin along the grid points.
        grid: BoundedGrid object
        distances: array of distances from the source point
        start: int
            Index of the starting point in the grid.
        step_size: float
            Step size for the gradient descent.
        """
        path = [grid.valid_points[start]]
        current_idx = start
        final_idx = np.argmin(distances)
        final_point = grid.valid_points[final_idx]
        while True:
            current = grid.valid_points[current_idx]
            if self.dist(current, final_point) < tol:
                break
            elif current_idx == final_idx:
                break
            neighbors = []
            for delta in it.product(*[[1, 0, -1], ]*grid.dim):
                if not np.any(delta):
                    continue
                neighbor_idx = grid.neighbor(current_idx, delta)
                if neighbor_idx != -1:
                    neighbors.append(neighbor_idx)
            if not neighbors:
                print("Warning: no neighbors found.")
                break
            next_idx = min(neighbors, key=lambda idx: distances[idx])
            if distances[next_idx] > distances[current_idx] + tol:
                break
            path.append(grid.valid_points[next_idx])
            current_idx = next_idx
        return np.array(path)