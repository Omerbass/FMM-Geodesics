import numpy as np

from geofinder import *  # noqa: F403
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
    
def main_antiferro_sivak(x0, show_plot=False):
    aFmetric = metrics.AntiFerroSivak()
    # grid = BoundedGrid(cartesian_boundaries=[(1.01, 6.3), (-7.5, 7.5)], deltas=[0.03, 0.03], dim=2, bound_function = aFmetric.is_ordered_phase)
    grid = BoundedGrid(cartesian_boundaries=[(1., 4), (-4.5, 4.5)], deltas=[0.005, 0.005], dim=2, bound_function = aFmetric.is_ordered_phase)

    positions = grid.valid_points

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()
    
    source = grid.point_to_idx(x0)

    geo = FMMGeodesicPaths(aFmetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    if show_plot:
        plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
        plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
        plt.colorbar()
        plt.show()

    np.savez(f"data/sivak_antiferro_geodesic_paths_b0={positions[source, 0]:.3f}_a0={positions[source, 1]:.3f}.npz", #_high-res
        positions=positions, distances=distances, source=source, triangles=triangles, grid=grid, deluanay=delaunay)

def main_bethe_antiferro(x0, show_plot=False):
    betheMetric = metrics.BetheIsing(z=3)
    grid = BoundedGrid(cartesian_boundaries=[(-3.0, -0.55), (-8.5, 8.5)], deltas=[0.005, 0.005],
                        dim=2, bound_function=betheMetric.is_ordered_phase)

    positions = grid.valid_points

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()

    source = grid.point_to_idx(x0)

    geo = FMMGeodesicPaths(betheMetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    if show_plot:
        plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5)
        plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
        plt.colorbar()
        plt.show()

    np.savez(f"data/bethe_antiferro_geodesic_paths_K0={positions[source, 0]:.3f}_h0={positions[source, 1]:.3f}.npz",
        positions=positions, distances=distances, source=source, triangles=triangles, grid=grid, deluanay=delaunay)

def _graded_1d_points(anchors, lo, hi, fine_delta, coarse_delta, band_half_width):
    """
    1D point set covering [lo, hi]: fine_delta resolution within band_half_width
    of any point in `anchors`, coarse_delta everywhere else. NaN anchors are
    skipped. Used to concentrate grid points near the phase transition line
    without paying for uniform fine resolution across the whole domain.
    """
    coarse = np.arange(lo, hi, coarse_delta)
    keep = np.ones(len(coarse), dtype=bool)
    parts = []
    for a in anchors:
        if np.isnan(a):
            continue
        f_lo, f_hi = max(lo, a - band_half_width), min(hi, a + band_half_width)
        if f_hi > f_lo:
            parts.append(np.arange(f_lo, f_hi, fine_delta))
        keep &= ~((coarse >= f_lo) & (coarse <= f_hi))
    parts.append(coarse[keep])
    return np.unique(np.round(np.concatenate(parts), 8))

def main_bethe_antiferro_transition(x0, show_plot=False, K_bounds=(-3.0, -0.55), dK=0.01,
                                     h_bounds=(-8.5, 8.5), fine_delta=0.005, coarse_delta=0.05,
                                     band_half_width=1.0):
    """
    Like main_bethe_antiferro, but spans both the Neel-ordered and disordered
    phases -- no BoundedGrid/is_ordered_phase filtering -- so geodesics can run
    *through* the phase transition instead of stopping at its boundary.

    Point density is graded per-row (see _graded_1d_points): fine_delta
    resolution within band_half_width of the transition lines
    h = +-phase_transition_line(K), coarse_delta everywhere else. This keeps
    the total point count manageable despite covering roughly twice the (K,h)
    area of main_bethe_antiferro (which only filled the ordered diamond).
    Since the per-row h-grid depends on K (through phase_transition_line(K)),
    the resulting positions are an unstructured point cloud, not a regular
    grid -- Delaunay only needs the raw points, so this is fine, but there is
    no BoundedGrid object to save/return (grid=None in the output), and no
    point_to_idx: the source index is picked by nearest point instead.
    """
    betheMetric = metrics.BetheIsing(z=3)
    Ks = np.arange(K_bounds[0], K_bounds[1], dK)
    hcs = np.array([betheMetric.phase_transition_line(K) for K in Ks])

    rows = []
    for K, hc in zip(Ks, hcs):
        hs = _graded_1d_points([hc, -hc], h_bounds[0], h_bounds[1], fine_delta, coarse_delta, band_half_width)
        rows.append(np.column_stack([np.full(hs.shape, K), hs]))
    positions = np.concatenate(rows, axis=0)

    delaunay = Delaunay(positions)
    triangles = delaunay.simplices.tolist()

    source = np.argmin(np.linalg.norm(positions - x0, axis=1))

    geo = FMMGeodesicPaths(betheMetric.metric, dim=2)

    distances = geo.fast_marching_method(positions, triangles, source)

    if show_plot:
        plt.scatter(positions[:, 0], positions[:, 1], c=distances, cmap='viridis', alpha=0.5, s=2)
        plt.scatter(positions[source, 0], positions[source, 1], c='red', s=10, label='Source')
        valid = ~np.isnan(hcs)
        plt.plot(Ks[valid], hcs[valid], 'w--', linewidth=1)
        plt.plot(Ks[valid], -hcs[valid], 'w--', linewidth=1)
        plt.colorbar()
        plt.show()

    np.savez(f"data/bethe_antiferro_transition_geodesic_paths_K0={positions[source, 0]:.3f}_h0={positions[source, 1]:.3f}.npz",
        positions=positions, distances=distances, source=source, triangles=triangles, grid=None, deluanay=delaunay)

if __name__ == "__main__":
    # main_antiferro_sivak(np.array([2.51, 1.5]))
    # main_bethe_antiferro(np.array([-0.9, 0.6]))
    main_bethe_antiferro_transition(np.array([-1.0, 1.95]), fine_delta=0.002, coarse_delta=0.01, band_half_width=0.5)


