import numpy as np
import heapq
from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
from rich.progress import Progress
import metrics
import warnings
import holoviews as hv
hv.extension('bokeh')
from bokeh.plotting import show  # noqa: E402

warnings.filterwarnings("error", category=RuntimeWarning)

class FMMGeodesicPaths:
    def __init__(self, metric, dim=2, **kwargs):
        self.metric = metric
        self.dim = dim
        for key, val in kwargs.items():
            setattr(self, key, val)

    def dist(self, a, b):
        """
        Compute the infenitesimal distance between two points.
        """
        return a.T @ self.metric((a+b)/2) @ b
    
    def geonorm(self, p, a):
        """
        Compute the norm of a vector, on the geodesic at point p.
        """
        return np.sqrt(a.T @ self.metric(p) @ a)
    
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

    def unfold_and_find_virtual_edge(self, positions, triangles, vertex_index):
        # Simplified unfolding: just selects a neighboring vertex as virtual support
        for tri in triangles:
            if vertex_index in tri:
                for idx in tri:
                    if idx != vertex_index:
                        return idx
        return None

    def update_triangle(self, T, positions, triangles, A, B, C, F=1.0):
        Ta, Tb, Tc = T[A], T[B], T[C]

        # Ensure Ta <= Tb
        if Ta > Tb:
            Ta, Tb = Tb, Ta
            A, B = B, A

        u = Tb - Ta
        
        vec_ab = positions[B] - positions[A]
        vec_ac = positions[C] - positions[A]

        a = self.geonorm(positions[A], vec_ab)
        b = self.geonorm(positions[A], vec_ac)
        cos_theta = self.geoip(positions[A], vec_ab, vec_ac) / (a * b)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # if is_obtuse_triangle(positions[A], positions[B], positions[C]):
        #     virtual_vertex = unfold_and_find_virtual_edge(positions, triangles, C)
        #     if virtual_vertex is not None:
        #         T_virtual = T[virtual_vertex] + np.linalg.norm(positions[C] - positions[virtual_vertex]) * F
        #         return min(Tc, T_virtual)

        t = self.solve_quadratic(a**2 + b**2 - 2*a*b*cos_theta,
                            2*b*u*(a*cos_theta - b),
                            b**2*(u**2 - (F**2)*(a**2)*(sin_theta**2)))
        try:
            if u < t and (a * cos_theta <  b * (1 - u/t)) and\
                    (b * (1 - u/t) < (a / cos_theta if cos_theta != 0 else np.inf)):
                Tc_new = Ta + t
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

        with Progress() as progress:
            task0 = progress.add_task("[cyan]heap")
            task1 = progress.add_task("[green]alive", total=len(status))
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

def main_sphere():
    positions = np.reshape(np.meshgrid(np.linspace(0.001, np.pi-0.001, 50),np.linspace(0, 2*np.pi, 50)), (2, -1)).T
    triangles = Delaunay(positions).simplices
    source = 1225
    print("source:", positions[source])

    geo = FMMGeodesicPaths(metrics.sphereMetric, dim=2)

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
    show(hv.render(plot))

    print(positions[-1], ":", distances[-1])

def main_antiferro():
    positions = np.reshape(np.meshgrid(np.linspace(0.001, np.pi-0.001, 20),np.linspace(0, 2*np.pi, 20)), (2, -1)).T
    triangles = Delaunay(positions).simplices
    source = 255
    print("source:", positions[source])

    afmetric = metrics.AntiFerro()
    geo = FMMGeodesicPaths(afmetric.metric(), dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)
    # plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    # plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
    # plt.colorbar()
    # plt.show()
    plot = (hv.HeatMap((positions[:, 0], positions[:, 1], distances), label='Geodesic Distances').opts(
        colorbar=True, cmap='viridis', tools=["hover",], xlabel='Theta', ylabel='Phi'
    ) * hv.Points((*positions[source],), label='Source').opts(color="red",size=10)).opts(
        legend_position='top_left', width=800, height=550, title="Geodesic Paths on Antiferro"
    )
    show(hv.render(plot))

if __name__ == "__main__":
    # Example usage (requires proper triangulation and positions):
    pass
