from bokeh.plotting import show as bkshow
import numpy as np
import heapq
from scipy.spatial import Delaunay
from rich.progress import Progress
import rich.progress as rprog
from rich import print as rprint # noqa: F401
import itertools as it
import warnings

from . import metrics
from .irregular_grids import BoundedGrid

import matplotlib.pyplot as plt
import holoviews as hv
hv.extension('bokeh')

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

    def update_triangle(self, T, positions, triangles, A, B, C, F=1.0):
        Ta, Tb, Tc = T[A], T[B], T[C]

        # Ensure Ta <= Tb
        if Ta > Tb:
            Ta, Tb = Tb, Ta
            A, B = B, A

        u = Tb - Ta
        
        if np.isinf(u):
            return np.inf

        vec_ab = positions[B] - positions[A]
        vec_ac = positions[C] - positions[A]

        a = self.geonorm(positions[A], vec_ab)
        b = self.geonorm(positions[A], vec_ac)
        cos_theta = self.geoip(positions[A], vec_ab, vec_ac) / (a * b)
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
        T = np.full(num_points, np.inf)
        status = ['far'] * num_points

        T[source] = 0.0
        status[source] = 'alive'

        heap = []
        for tri in triangles:
            if source in tri:
                for p in tri:
                    if p != source and status[p] == 'far':
                        status[p] = 'close'
                        T[p] = self.dist(positions[p], positions[source])
                        heapq.heappush(heap, (T[p], p))

        with Progress(*Progress.get_default_columns(), rprog.TimeElapsedColumn(), rprog.MofNCompleteColumn()) as progress:
            task0 = progress.add_task("[cyan]heap", total=len(status))
            task1 = progress.add_task("[white]alive", total=len(status))
            task2 = progress.add_task("[yellow]close", total=len(status))
            while heap:
                _, p = heapq.heappop(heap)
                status[p] = 'alive'
                progress.update(task0, completed=len(heap))
                progress.update(task1, completed=status.count('alive'))
                progress.update(task2, completed=status.count('close'))
                neighbor_tris = [tri for tri in triangles if p in tri]

                for tri in neighbor_tris:
                    for q in tri:
                        if status[q] != 'alive':
                            r = [v for v in tri if v not in [p, q]][0]
                            if status[r] == 'alive':
                                T_old = T[q]
                                T[q] = self.update_triangle(T, positions, triangles, p, r, q)

                                if status[q] == 'far':
                                    heapq.heappush(heap, (T[q], q))
                                    status[q] = 'close'
                                elif T[q] < T_old:
                                    heapq.heappush(heap, (T[q], q))

        return T

    def gradient_descent_to_origin_along_simplices(self, grid, delaunay, distances, start, max_steps=1000, tol=1e-6):
        path = [grid.valid_points[start]]
        current = grid.valid_points[start]
        final_idx = np.argmin(distances)
        final_point = grid.valid_points[final_idx]

        def get_normalized_gradient(point, simplex):
            if simplex == -1:
                print(point)
                print("Warning: point outside of triangulation.")
                return np.zeros(self.dim)
            vertices = delaunay.simplices[simplex]
            d1, d2, d3 = distances[vertices]
            p1, p2, p3 = grid.valid_points[vertices]
            p_mat = np.column_stack((p2 - p1, p3 - p1))
            d_vec = np.array([d2 - d1, d3 - d1])
            # grad_embedded = p_mat @ np.linalg.solve(p_mat.T @ p_mat, d_vec)
            # grad = np.einsum("ij, j", self.inv_metric(point), grad_embedded)
            # grad_norm = self.geonorm(point, grad)
            grad = p_mat @ np.linalg.solve(p_mat.T @ p_mat, d_vec)
            grad_norm = np.linalg.norm(grad)
            return grad / grad_norm

        def intersect_lines(p1, p2, v1, v2):
            A = np.array([v1, -v2]).T
            if np.abs(np.linalg.det(A)) < 1e-12:
                return None
            b = p2 - p1
            t = np.linalg.solve(A, b)
            return (p1 + t[0] * v1, t[0], t[1])

        current_simplex = delaunay.find_simplex(current)

        for _ in range(max_steps):
            if final_idx in delaunay.simplices[current_simplex]:
                path.append(final_point)
                break
            elif self.dist(current, final_point) < tol:
                break
            grad = get_normalized_gradient(current, current_simplex)
            if not np.any(grad):
                current = grid.valid_points[grid.point_to_idx(current)]
                path.append(current)
                rprint(f"[red]Warning: {current} outside of grid.")
                return np.array(path)
            print("----", current, "----")
            print("triangle :", delaunay.simplices[current_simplex])
            print("gradient :", grad)
            for i1, i2 in it.combinations([0, 1,2], 2):
                ix1, ix2 = delaunay.simplices[current_simplex][[i1, i2]]
                p1, p2 = grid.valid_points[[ix1, ix2]]
                edge_vec = p2 - p1
                res = intersect_lines(current, p1, -grad, edge_vec)
                print("intersection :", res, " with edge ", p1, p2)
                if res is None:
                    continue
                intersection, t1, t2 = res
                if t1 > 1e-12 and 0 <= t2 <= 1:
                    current = intersection
                    path.append(current)
                    print(delaunay.neighbors[current_simplex], i1, i2)
                    current_simplex = delaunay.neighbors[current_simplex][list({0,1,2} - {i1, i2})[0]]
                    print("next simplex :", current_simplex, ", edges:", delaunay.simplices[current_simplex])
                    break
            else:
                rprint(f"[red]Warning: no intersection found for point {current}, stopping.")
                break

        return np.array(path)

    def gradient_descent_to_origin(self, grid, delaunay, distances, start, step_size=0.01, max_steps=1000, tol=1e-6):
        path = [grid.valid_points[start]]
        current = grid.valid_points[start]
        final_idx = np.argmin(distances)
        final_point = grid.valid_points[final_idx]

        # print("no inverse metric correction")

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

        with Progress() as progress:
            task = progress.add_task("[cyan]distance", total=distances[start])
            for _ in range(max_steps):
                true_step = np.min((step_size, step_size / tol / 10 * (self.dist(current, final_point))))
                grad = get_normalized_gradient(current)
                if not np.any(grad):
                    current = grid.valid_points[grid.point_to_idx(current)]
                    path.append(current)
                    rprint(f"[red]Warning: {current} outside of grid.")
                    break
                # RK2
                next_pred = current - true_step * grad
                grad_pred = get_normalized_gradient(next_pred)

                nxt = current - 0.5 * true_step * (grad + grad_pred)

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

def main_flat():
    positions = np.reshape(np.meshgrid(np.linspace(0.001, np.pi-0.001, 100),np.linspace(0, 2*np.pi, 100)), (2, -1)).T
    delaunay = Delaunay(positions)
    triangles = delaunay.simplices
    source = 4851
    print("source:", positions[source])

    geo = FMMGeodesicPaths(lambda x: np.eye(2), dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)
    # print("Distances:", distances)
    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    plt.colorbar()
    plt.show()
    # plot = (hv.HeatMap((positions[:, 0], positions[:, 1], distances), label='Geodesic Distances').opts(
    #     colorbar=True, cmap='viridis', tools=["hover",], xlabel='Theta', ylabel='Phi'
    # ) * hv.Points((*positions[source],), label='Source').opts(color="red",size=10)).opts(
    #     legend_position='top_left', width=800, height=550, title="Geodesic Paths on Sphere"
    # )
    # bkshow(hv.render(plot))

    print(positions[-1], ":", distances[-1])
    np.savez(f"data/flat_geodesic_paths_x0=({positions[source][0]:.3f}, {positions[source][1]:.3f}).npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=None, deluanay=delaunay)


def main_sphere():
    positions = np.reshape(np.meshgrid(np.linspace(0.001, np.pi-0.001, 100),np.linspace(0, 2*np.pi, 100)), (2, -1)).T
    delaunay = Delaunay(positions)
    triangles = delaunay.simplices
    source = 4851
    print("source:", positions[source])

    geo = FMMGeodesicPaths(metrics.Sphere().metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)
    # print("Distances:", distances)
    # plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    # plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    # plt.colorbar()
    # plt.show()
    plot = (hv.HeatMap((positions[:, 0], positions[:, 1], distances), label='Geodesic Distances').opts(
        colorbar=True, cmap='viridis', tools=["hover",], xlabel='Theta', ylabel='Phi'
    ) * hv.Points((*positions[source],), label='Source').opts(color="red",size=10)).opts(
        legend_position='top_left', width=800, height=550, title="Geodesic Paths on Sphere"
    )
    bkshow(hv.render(plot))

    print(positions[-1], ":", distances[-1])
    np.savez(f"data/sphere_geodesic_paths_x0=({positions[source][0]:.3f}, {positions[source][1]:.3f}).npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=None, deluanay=delaunay)

def main_antiferro_old():
    N = 200
    t = np.linspace((0.1, )*N, (1-0.01, )*N, N)
    h = np.array([ np.linspace(-(T/2 * np.log((1+np.sqrt(1-T))/(1-np.sqrt(1-T))) + np.sqrt(1-T))*(1 - 1e-4),
        (T/2 * np.log((1+np.sqrt(1-T))/(1-np.sqrt(1-T))) + np.sqrt(1-T))*(1 - 1e-4), N) for T in t[:,0]])
    positions = np.reshape((t,h), (2, -1)).T
    delaunay = Delaunay(positions)
    triangles = delaunay.simplices
    source = 10500
    print("source:", positions[source])

    afmetric = metrics.AntiFerro()
    geo = FMMGeodesicPaths(afmetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    np.savez(f"antiferro_geodesic_paths_T0={positions[source, 0]}_h0={positions[source, 1]}.npz", 
        positions=positions, distances=distances, source=source, delaunay=delaunay, triangles=triangles)

    # plot = (hv.HeatMap((positions[:, 0], positions[:, 1], distances), label='Geodesic Distances').opts(
    #     colorbar=True, cmap='viridis', tools=["hover",], xlabel='Theta', ylabel='Phi'
    # ) * hv.Points((*positions[source],), label='Source').opts(color="red",size=10)).opts(
    #     legend_position='top_left', width=800, height=550, title="Geodesic Paths on Antiferro Mean Field"
    # )
    # bkshow(hv.render(plot))

    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    plt.colorbar()
    plt.show()

def main_antiferro_extra_triangles():
    aFmetric = metrics.AntiFerro()
    grid = BoundedGrid(cartesian_boundaries=[(0.1, 0.999), (-1.2, 1.2)], deltas=[0.003, 0.003], dim=2, bound_function = aFmetric.is_ordered_phase)

    positions = grid.valid_points

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()

    additional_triangles = []    
    with Progress() as progress:
        task = progress.add_task("[red]Adding triangles", total=len(positions))
        for idx, point in enumerate(positions):
            progress.update(task, advance=1)
            met = aFmetric.metric(point)
            if not np.isclose(np.linalg.det(met), 0, atol = 1e-5, rtol=0):
                P = np.abs(met[0,1]) / met[0,0]
                Q = met[1,1] / np.abs(met[0,1])
                if P >= 1:
                    p = P%1
                    q = Q - P+p
                else:
                    p = P
                    q = Q
                n = 1
                while True:
                    Pn = p*n
                    Qn = q*n
                    m = np.ceil(Pn)
                    if m < Qn:
                        break
                    n += 1
                if P >= 1:
                    m += np.floor(P)*n
                if met[0,1] > 0:
                    n = -n
                new_triangles = [ tri for tri in ((idx, grid.neighbor(idx, np.array((n, m), dtype=int)), grid.neighbor(idx, np.array((np.sign(n), 0), dtype=int))),
                    (idx, grid.neighbor(idx, np.array((n, m), dtype=int)), grid.neighbor(idx, np.array((0, np.sign(m)), dtype=int))),
                    (idx, grid.neighbor(idx, np.array((-n, -m), dtype=int)), grid.neighbor(idx, np.array((-np.sign(n), 0), dtype=int))),
                    (idx, grid.neighbor(idx, np.array((-n, -m), dtype=int)), grid.neighbor(idx, np.array((0, -np.sign(m)), dtype=int)))) if -1 not in tri]
                    
                additional_triangles.extend(new_triangles)
    
    rprint(f"[green]Added {len(additional_triangles)} additional triangles ({100*len(additional_triangles)/len(triangles):.2f}%)")

    triangles.extend(additional_triangles)

    source = grid.point_to_idx(np.array([0.6, 1.]))

    geo = FMMGeodesicPaths(aFmetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    plt.colorbar()
    plt.show()

    np.savez(f"data/antiferro_geodesic_paths_T0={positions[source, 0]:.3f}_h0={positions[source, 1]:.3f}.npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=grid, deluanay=delaunay)

def main_antiferro():
    aFmetric = metrics.AntiFerro()
    grid = BoundedGrid(cartesian_boundaries=[(0.1, 0.999), (-1.2, 1.2)], deltas=[0.003, 0.003], dim=2, bound_function = aFmetric.is_ordered_phase)

    positions = grid.valid_points

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()
    # additional_triangles = []    
    # with Progress() as progress:
    #     task = progress.add_task("[red]Adding triangles", total=len(positions))
    #     for idx, point in enumerate(positions):
    #         progress.update(task, advance=1)
    #         met = aFmetric.metric(point)
    #         if not np.isclose(np.linalg.det(met), 0, atol = 1e-5, rtol=0):
    #             P = np.abs(met[0,1]) / met[0,0]
    #             Q = met[1,1] / np.abs(met[0,1])
    #             if P >= 1:
    #                 p = P%1
    #                 q = Q - P+p
    #             else:
    #                 p = P
    #                 q = Q
    #             n = 1
    #             while True:
    #                 Pn = p*n
    #                 Qn = q*n
    #                 m = np.ceil(Pn)
    #                 if m < Qn:
    #                     break
    #                 n += 1
    #             if P >= 1:
    #                 m += np.floor(P)*n
    #             if met[0,1] > 0:
    #                 n = -n
    #             new_triangles = [ tri for tri in ((idx, grid.neighbor(idx, np.array((n, m), dtype=int)), grid.neighbor(idx, np.array((np.sign(n), 0), dtype=int))),
    #                 (idx, grid.neighbor(idx, np.array((n, m), dtype=int)), grid.neighbor(idx, np.array((0, np.sign(m)), dtype=int))),
    #                 (idx, grid.neighbor(idx, np.array((-n, -m), dtype=int)), grid.neighbor(idx, np.array((-np.sign(n), 0), dtype=int))),
    #                 (idx, grid.neighbor(idx, np.array((-n, -m), dtype=int)), grid.neighbor(idx, np.array((0, -np.sign(m)), dtype=int)))) if -1 not in tri]
                    
    #             additional_triangles.extend(new_triangles)
    
    # rprint(f"[green]Added {len(additional_triangles)} additional triangles ({100*len(additional_triangles)/len(triangles):.2f}%)")

    # triangles.extend(additional_triangles)

    source = grid.point_to_idx(np.array([0.6, 1.]))

    geo = FMMGeodesicPaths(aFmetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    plt.colorbar()
    plt.show()

    np.savez(f"data/antiferro_geodesic_paths_T0={positions[source, 0]:.3f}_h0={positions[source, 1]:.3f}.npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=grid, deluanay=delaunay)
    
def main_antiferro_sivak():
    aFmetric = metrics.AntiFerroSivak()
    grid = BoundedGrid(cartesian_boundaries=[(1.01, 6.3), (-7.5, 7.5)], deltas=[0.03, 0.03], dim=2, bound_function = aFmetric.is_ordered_phase)

    positions = grid.valid_points

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()
    
    source = grid.point_to_idx(np.array([4, 3.5]))

    geo = FMMGeodesicPaths(aFmetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    plt.colorbar()
    plt.show()

    np.savez(f"data/sivak_antiferro_geodesic_paths_T0={positions[source, 0]:.3f}_h0={positions[source, 1]:.3f}.npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=grid, deluanay=delaunay)

if __name__ == "__main__":
    main_antiferro_sivak()
