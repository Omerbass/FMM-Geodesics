import numpy as np
import heapq
from scipy.spatial import Delaunay  # noqa: F401
import matplotlib.pyplot as plt
from rich.progress import Progress


def solve_quadratic(a, b, c):
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return np.inf
    return (-b + np.sqrt(disc)) / (2 * a)

def is_obtuse_triangle(a, b, c):
    edges = [np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)]
    edges.sort()
    return edges[2]**2 > edges[0]**2 + edges[1]**2

def unfold_and_find_virtual_edge(positions, triangles, vertex_index):
    # Simplified unfolding: just selects a neighboring vertex as virtual support
    for tri in triangles:
        if vertex_index in tri:
            for idx in tri:
                if idx != vertex_index:
                    return idx
    return None

def update_triangle(T, positions, triangles, A, B, C, F=1.0):
    Ta, Tb, Tc = T[A], T[B], T[C]

    # Ensure Ta <= Tb
    if Ta > Tb:
        Ta, Tb = Tb, Ta
        A, B = B, A

    u = Tb - Ta
    
    vec_ab = positions[B] - positions[A]
    vec_ac = positions[C] - positions[A]
    
    a = np.linalg.norm(vec_ab)
    b = np.linalg.norm(vec_ac)
    cos_theta = np.dot(vec_ab, vec_ac) / (a * b)
    sin_theta = np.sqrt(1 - cos_theta**2)

    # if is_obtuse_triangle(positions[A], positions[B], positions[C]):
    #     virtual_vertex = unfold_and_find_virtual_edge(positions, triangles, C)
    #     if virtual_vertex is not None:
    #         T_virtual = T[virtual_vertex] + np.linalg.norm(positions[C] - positions[virtual_vertex]) * F
    #         return min(Tc, T_virtual)

    t = solve_quadratic(a**2 + b**2 - 2*a*b*cos_theta,
                        2*b*u*(a*cos_theta - b),
                        b**2*(u**2 - (F**2)*(a**2)*(sin_theta**2)))

    if u < t and \
            (a * cos_theta <  (b * (t - u)) / t) and\
            ((b * (t - u)) / t < (a / cos_theta if cos_theta != 0 else np.inf)):
        Tc_new = Ta + t
    else:
        Tc_new = min(Tc, Ta + b * F, Tb + np.linalg.norm(positions[C]-positions[B]) * F)

    return min(Tc, Tc_new)

def fast_marching_method(positions, triangles, source):
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
                    T[p] = np.linalg.norm(positions[p] - positions[source])
                    heapq.heappush(heap, (T[p], p))

    with Progress() as progress:
        task0 = progress.add_task("[cyan]heap")
        task1 = progress.add_task("[green]alive", total=len(status)-1)
        task2 = progress.add_task("[yellow]close", total=len(status)-1)
        while heap:
            _, p = heapq.heappop(heap)
            status[p] = 'alive'
            progress.update(task0, completed=len(heap))
            progress.update(task1, advance=1)
            progress.update(task2, completed=status.count('close'))
            neighbor_tris = [tri for tri in triangles if p in tri]

            for tri in neighbor_tris:
                for q in tri:
                    if status[q] != 'alive':
                        r = [v for v in tri if v not in [p, q]][0]
                        if status[r] == 'alive':
                            T_old = T[q]
                            T[q] = update_triangle(T, positions, triangles, p, r, q)

                            if status[q] == 'far':
                                heapq.heappush(heap, (T[q], q))
                                status[q] = 'close'
                            elif T[q] < T_old:
                                heapq.heappush(heap, (T[q], q))

    return T

if __name__ == "__main__":
    # Example usage (requires proper triangulation and positions):
    positions = np.reshape(np.meshgrid(np.linspace(0, 1, 50),np.linspace(0, 1, 50)), (2, -1)).T
    triangles = Delaunay(positions).simplices
    source = 0

    distances = fast_marching_method(positions, triangles, source)
    # print("Distances:", distances)
    plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.show()
    print(positions[-1], ":", distances[-1])
