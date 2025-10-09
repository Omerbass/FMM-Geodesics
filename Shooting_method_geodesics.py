import itertools
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
# from p_tqdm import p_map
from inspect import signature
from typing import Callable #, Iterable, Union
import warnings
import resource
from functools import wraps
import metrics

def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator

class ShootingMethodGeodesics:
    def __init__(self, metric, christoffel_func, dim):
        self.metric = metric
        self.christoffel_func = christoffel_func
        self.dim = dim

    def dist(self, path):
        """
        Compute the distance of a path.

        Parameters:
        path (array-like): The path. Assumed to be dense enough.

        Returns:
        float: The distance of the path.
        """
        metric = [self.metric(x) for x in (path[1:, :] + path[:-1, :]) / 2]
        diffs = np.diff(path, axis=0)
        return np.sum([np.sqrt(d.T @ m @ d) for d,m in zip(diffs, metric)])


    def geodesic_equation(self, t, y):
        """y = [x^i, v^i] where v^i = dx^i/dt
        assumes y:=(x, v) where x is the position and v is the velocity"""
        x, v = y[:self.dim], y[self.dim:]
        Gamma = self.christoffel_func(x)
        dvdt = -np.einsum('ijk,j,k->i', Gamma, v, v)
        assert not np.any(np.isnan(x)), "x contains NaN values"
        assert not np.any(np.isnan(v)), "v contains NaN values"
        if  np.any(np.isnan(dvdt)):
            return(np.concatenate([v, np.zeros_like(v)]))
        return np.concatenate([v, dvdt])

    def geodesic_equation_add_total_length(self, t, y):
        """y = [x^i, v^i] where v^i = dx^i/dt
        assumes y:=(x, v, total_length) where x is the position, v is the velocity and total_length is the accumulated length"""
        x, v = y[:self.dim], y[self.dim:-1]
        step = np.concatenate([self.geodesic_equation(t, y[:-1]), [np.sqrt(np.einsum("ij,i,j", self.metric(x), v, v))]])
        # print(step)
        return step

    def circular_limits(self, y):
        return y
    
    @eventAttr()
    def hard_limits(self, t, y, *args):
        return 1
    
    def path(self, x0, alpha, dist=1000, tol=1e-5, v0 = 1):
        """Find the geodesic path from x0 to x1"""
        stopevent = lambda t, y, *args: dist - y[-1] + 1e-5  # noqa: E731
        stopevent.terminal = True
        y0 = np.concatenate([x0, [v0*np.cos(alpha), v0*np.sin(alpha), 0]])
        sol = solve_ivp(self.geodesic_equation_add_total_length, (0, dist*20), y0, max_step=tol*0.5, events=(stopevent, self.hard_limits, ))
        path = self.circular_limits(sol.y[:self.dim,:])
        return path

    # shooting + compartmentalizing
    def shooting_and_comp(self, x0, x1, tol=1e-2):
        """Find the initial velocity that connects x0 to x1"""
        dim = self.dim
        straight_path = np.linspace(x0, x1, 100)
        straight_dist = np.sum([np.sqrt((x-y).T @ self.metric((x+y)/2) @ (x-y)) for x,y in zip(straight_path[1:], straight_path[:-1])])
        stopevent = lambda t, y, *args: straight_dist * 1.02 - y[-1]  # noqa: E731
        stopevent.terminal = True
        
        def objective(alpha):
            # Solve the geodesic equation with initial conditions
            y0 = np.concatenate([x0, [np.cos(alpha), np.sin(alpha)], [0]])
            sol = solve_ivp(self.geodesic_equation_add_total_length, (0, straight_dist*20), y0, 
                            max_step=tol*0.5, events=(stopevent, self.hard_limits ))
            xs = self.circular_limits(sol.y[:dim, :])
            # print(np.linalg.norm(xs.T - x1, axis=1))
            
            # Return the error (distance to target)
            return np.min(np.linalg.norm((xs.T - x1).T.T, axis=1))

        def shots(objective, alphas):
            mindist = list(map(objective, alphas)) #, tqdm=tqdm
            return np.argmin(mindist), min(mindist)

        alpharange = (0, 2*np.pi)
        # mindist=tol+1
        N = 50
        for _ in range(6):
            alphas = np.linspace(alpharange[0], alpharange[1], N+1)
            dalpha = (alpharange[1] - alpharange[0])/N
            print("alpharange:", np.rad2deg(alpharange))
            ix, mindist = shots(objective, alphas)
            alphamin = alphas[ix]
            alpharange = (alphamin-dalpha, alphamin+dalpha)
            # print(mindist, ":", np.rad2deg(alpharange), np.rad2deg(alphamin))
            if mindist<tol:
                break
            elif dalpha < 1e-4:
                warnings.warn(f"Could not converge to the target within tolerance {tol}. Best distance: {mindist}")
                break
    
        y0 = np.concatenate([x0, [np.cos(alphamin), np.sin(alphamin)], [0]])
        sol = solve_ivp(self.geodesic_equation_add_total_length, (0, straight_dist*20), y0, 
                        max_step=tol*0.5, events=(stopevent, ))

        ixf = np.argmin(np.linalg.norm(sol.y[:self.dim,:].T-x1, axis=1))
        geodesic_dist = sol.y[-1, ixf]
        return alphamin, geodesic_dist,{"mindist": mindist, "alpharange": alpharange, "ixf": ixf, "sol": sol}

    def shooting_method(self, x0, x1, tol=1e-2):
        """Find the geodesic path from x0 to x1"""
        alpha, geodesic_dist, meta = self.shooting_and_comp(x0, x1, tol)
        path = meta["sol"].y[:self.dim, :meta["ixf"]+1]

        return {"path": path, "α0": alpha, "dist": geodesic_dist, "meta": meta}
    
    __call__ = shooting_method

class SivakShooting(ShootingMethodGeodesics):
    def __init__(self):
        self.dim = 2
        self.z = 1
        sivakAF = metrics.AntiFerroSivak()
        self.metricspace = sivakAF
        self.metric = sivakAF.metric
        self.christoffel_func = sivakAF.christoffel_func
    
    @eventAttr()
    def hard_limits(self, t, y, *args):
        α_max = self.metricspace.phase_transition_line(y[0])
        if y[0] < self.z * 1.03 or np.abs(y[1]) * 0.97 > α_max or y[0]>6.5:
            print("reached hard limit, stopping integration\ty =", y)
            return -1
        return 1

def main_Sivak():
    sivak = SivakShooting()
    x0 = np.array([1.5, -0.5])
    x1 = np.array([3, 1])
    result = sivak.shooting_method(x0, x1, tol=1e-2)
    path = result["path"]
    α0 = result["α0"]
    dist = result["dist"]
    meta = result["meta"]
    print(f"Found geodesic from {x0} to {x1} with initial angle {np.rad2deg(α0):.2f}° and distance {dist:.2f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(path[0, :], path[1, :], 'r-', label='Geodesic Path', linewidth=2)
    ax.plot(x0[0], x0[1], 'go', label='Start Point', markersize=10)
    ax.plot(x1[0], x1[1], 'bo', label='End Point', markersize=10)
    ax.set_title('Geodesic Path in Sivak Metric Space')
    ax.set_xlabel('β')
    ax.set_ylabel('α')
    ax.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main_Sivak()