import numpy as np
from geofinder import SivakShooting # type: ignore
from matplotlib import pyplot as plt

def main_Sivak():
    sivak = SivakShooting()
    x0 = np.array([2.51, 1.5])
    x1 = np.array([1.49, 0.51])
    # x0 = np.array([3, sivak.metricspace.phase_transition_line(3)*0.9])
    result = sivak.shooting_method(x0, x1, tol=1e-2)
    path = result["path"]
    α0 = result["α0"]
    dist = result["dist"]
    # meta = result["meta"]
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

def main_shoot_once():
    sivak = SivakShooting()
    x0 = np.array([2.510, 1.5])
    x1 = np.array([1.0989011 , -0.63918993])
    α0 = np.deg2rad(-50)  # Initial angle in radians
    print(f"Shooting once from {x0} with angle {np.rad2deg(α0):.2f}°")
    
    path = sivak.path(x0, α0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(path[0, :], path[1, :], 'r-', label='Shot Path', linewidth=2)
    ax.plot(x0[0], x0[1], 'go', label='Start Point', markersize=10)
    ax.plot(x1[0], x1[1], 'bo', label='End Point', markersize=10)
    ax.set_title('Shooting Method Path in Sivak Metric Space')
    ax.set_xlabel('β')
    ax.set_ylabel('α')
    ax.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main_Sivak()
    # main_shoot_once()