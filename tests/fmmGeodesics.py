import numpy as np

from geofinder import *
from scipy.spatial import Delaunay

from rich import print as rprint
from rich.progress import Progress

import matplotlib.pyplot as plt
from bokeh.plotting import show as bkshow
import holoviews as hv
hv.extension('bokeh')

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
