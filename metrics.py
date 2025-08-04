import numpy as np
import scipy as sp
import warnings
from typing import Callable

def fixed_point(func:Callable, start:float, args=(), xtol:float=1e-9, maxiter:int=5000, method="iteration"):
    """
    Find the fixed point of a function using iteration.

    Parameters:
    func (callable): The function for which to find the fixed point.
    start (float): The initial guess for the fixed point.
    args (iterable): arguments to be passed to function.
    xtol (float, optional): The tolerance for convergence. Default is 1e-9.
    max_iter (int, optional): The maximum number of iterations. Default is 5000.

    Returns:
    float: The fixed point of the function.
    """
    if method != "iteration":
        raise NotImplementedError
    x0 = start
    for itr in range(maxiter):
        x1 = func(x0, *args)
        if np.max(abs(x1 - x0)) < xtol:
            break
        if np.any(np.isnan(x1)):
            raise Exception(f"got NaN in fixed point. last x was {x0}, args={args}")
        x0 = x1
    if itr == maxiter - 1:
        warnings.warn(f"Tolerance not reached\nachieved tolerance = {np.max(abs(func(x0, *args) - x0)):.3e} >= {xtol} = required tolerance")
    return x1

def sphereMetric(p):
        """
        Compute the metric tensor for a sphere at a given point.
        Parameters:
        -----------
        p : array-like
            A 2-dimensional vector of the form (theta, \phi ).
        Returns:
        --------
        numpy.ndarray
            A 2x2 diagonal matrix representing the metric tensor at the given point.
            The diagonal elements are equal to the square of the radius of the sphere,
            where the radius is the Euclidean norm of the input point `p`.
        """
        return np.diag([1, np.sin(p[0])**2])


class AntiFerro:
    dim = 2
    z = 1

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

    #                                        x=(T,h)
    def free_energy_non_minimized(self, m_s, x, z=1):
        T,h = x
        return 0.5*(z*(m_s[0]*m_s[1]) - h*(m_s[0] + m_s[1]) + 0.5 * T *(
        sp.special.xlogy(1+m_s[0], 1+m_s[0]) + sp.special.xlogy(1-m_s[0], 1-m_s[0]) +
        sp.special.xlogy(1+m_s[1], 1+m_s[1]) + sp.special.xlogy(1-m_s[1], 1-m_s[1]) )
        )

    #                       x=(T,h) 
    def tranceqn(self, m_s, x, z=1):
        m1 = np.tanh( (x[1] - self.z * m_s[1])/x[0] )
        m2 = np.tanh( (x[1] - self.z * m_s[0])/x[0] )
        return np.array([m1, m2])

    #                           x=(T,h) 
    def get_m_sublattices(self, x, grid=500):
        assert np.nan not in x, "x contains NaN values"
        # print(x)
        M1, M2 = np.meshgrid(*np.linspace([-1+1e-3,-1+1e-3],[1-1e-3,1-1e-3],grid).T)
        f = self.free_energy_non_minimized((M1,M2), x,z=self.z)
        ix, iy = np.unravel_index(np.argmin(f), f.shape)
        m1_0, m2_0 = M1[ix, iy], M2[ix,iy]
        if m1_0 == m2_0:
            m1_0 = np.min([m1_0 + 1e-2, 0.999])
            m2_0 = np.max([m2_0 - 1e-2, -0.999])
        # print("m0:", (m1_0, m2_0))
        m_s = fixed_point(self.tranceqn, (m1_0,m2_0), args=(x,self.z))
        return m_s

    #                x=(T,h)
    def metric(self, x):
        z = self.z
        T,h = x
        m1, m2 = self.get_m_sublattices(x)
        one_minus_m1_sq = 1 - m1**2
        one_minus_m2_sq = 1 - m2**2
        

        g_TT = (T*((one_minus_m1_sq)*np.arctanh(m1)**2 + (one_minus_m2_sq)*np.arctanh(m2)**2) +
            2*z*(one_minus_m1_sq)*(one_minus_m2_sq)*np.arctanh(m1)*np.arctanh(m2))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_Th = (-(T + z*one_minus_m1_sq)*one_minus_m2_sq*np.arctanh(m2) - (T + z*one_minus_m2_sq)*one_minus_m1_sq*np.arctanh(m1))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_hh = (T*(-m1**2 - m2**2 + 2) + 2*z*(one_minus_m1_sq)*(one_minus_m2_sq))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        return np.array([[g_TT, g_Th], [g_Th, g_hh]])

    def inv_metric(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanhm1 = np.arctanh(m1)
        atanhm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1

        sec_diag = -2*(T - m2_sq_minus1*z)*atanhm1/m2_sq_minus1 - 2*(T - (m1**2 -1)*z)*atanhm2/m1_sq_minus1
        return np.array([[-2*T*(m1_sq_minus1 + m2_sq_minus1)/(m1_sq_minus1*m2_sq_minus1) + 4*z, 
                           sec_diag], 
                          [sec_diag, 
                           -2*T*atanhm1**2/m2_sq_minus1 - 2*T*atanhm2**2/m1_sq_minus1 + 4*z*atanhm1*atanhm2]]) /(atanhm1 - atanhm2)**2

    def christoffel_func(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanhm1 = np.arctanh(m1)
        atanhm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1

        Γ_T_xx = [[(2*T*(T**2*m1 - m2*z**2*m1_sq_minus1**2)*atanhm1**3 + (-T**3*(3*m1**2 + m2**2 - 4)/m1_sq_minus1 + T*z**2*(m1_sq_minus1 - m2_sq_minus1)*m2_sq_minus1 + 
                    2*T*(T**2*m2 - m1*z**2*m2_sq_minus1**2)*atanhm2 + 4*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                    (-T**3*(m1**2 + 3*m2**2 - 4)/m2_sq_minus1 - T*z**2*(m1_sq_minus1 - m2_sq_minus1)*m1_sq_minus1 + 
                     2*T*(-T**2*m1 + 2*T*z*(m1 - m2)*(m1*m2 + 1) + m2*z**2*m1_sq_minus1**2)*atanhm2 + 
                     4*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + 2*(2*T**3 + 3*T**2*z*(m1_sq_minus1 + m2_sq_minus1) - 6*T*z**2*m1_sq_minus1*m2_sq_minus1 + 
                    T*(-T**2*m2 + 2*T*z*(-m1 + m2)*(m1*m2 + 1) + m1*z**2*m2_sq_minus1**2)*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm1*atanhm2)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2), 
                   
          (-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - 
           (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m1_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - 
            2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
            (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + 5*m2_sq_minus1) + 
                2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/
                (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2)],
        
                  [(-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*
                    (T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m1_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - 
                    2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
                    (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m2_sq_minus1 + m1_sq_minus1) + 
                    2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),

                   -(T**3*(m1_sq_minus1 + m2_sq_minus1)**2/(m1_sq_minus1*m2_sq_minus1) - 6*T**2*z*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*(m1**4 + 2*m1**2*(5*m2_sq_minus1 - 1) + m2**4 - 12*m2**2 + 12) +
                      2*T*(m1 - m2)*(-atanhm1 + atanhm2)*(T**2 + 2*T*z*(m1*m2 + 1) - m1*m2*z**2*(m1**2 + m1*m2 + m2**2 - 2) + z**2) - 2*z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))/
                      (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        
        Γ_h_xx = [[(-2*T*m2*z**2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*m1_sq_minus1**2*atanhm1**4 + 
                    (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*m2_sq_minus1*(T**3 + T**2*z*m1_sq_minus1 + 2*T*m1*z**2*m1_sq_minus1*m2_sq_minus1*atanhm2 +
                    T*z**2*m1_sq_minus1*m2_sq_minus1 - 3*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm2**3/m1_sq_minus1 + (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*
                    (-T**3 + 7*T**2*z*m2_sq_minus1 - 5*T*z**2*m1_sq_minus1*m2_sq_minus1 + 2*T*(T**2*m2 + 2*T*m1*z*m2_sq_minus1 + m1*z**2*m2_sq_minus1**2)*atanhm2 - 
                    z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm1*atanhm2**2 - (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3 - 7*T**2*z*m1_sq_minus1 + 
                    2*T**2*(m1 + m2)*(T + 2*m1*m2*z - 2*z)*atanhm2 + 5*T*z**2*m1_sq_minus1*m2_sq_minus1 + z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2*atanhm2 - 
                    (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-2*T*m2_sq_minus1*(T**2*m1 + 2*T*m2*z*m1_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 + 
                    m1_sq_minus1*(T**3 + T**2*z*m2_sq_minus1 + T*z**2*m1_sq_minus1*m2_sq_minus1 - 3*z**3*m1_sq_minus1*m2_sq_minus1**2))*atanhm1**3/m2_sq_minus1)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),
                   
                   (-2*T*m2*z*m1_sq_minus1*(T - m1**2*z + z)*atanhm1**3 + 2*T*(2*(T - m1**2*z + z)*(T - m2**2*z + z) +
                   (T**2*(m1 - m2) + T*m2*z*m1_sq_minus1 - m1*z**2*m2_sq_minus1**2)*atanhm2)* atanhm1*atanhm2 + 
                    (T**3*(-m1**2 + m2**2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*m2_sq_minus1*(T - m2**2*z + z)*atanhm2 + 
                     T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) - 2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                    (T**3*(m1_sq_minus1-m2_sq_minus1)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) - 
                     2*T*(T**2*(m1 - m2) - T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 - 
                     2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)],
                  
                  [(-2*T*m2*z*m1_sq_minus1*(T - m1**2*z + z)*atanhm1**3 + 2*T*(2*(T - m1**2*z + z)*(T - m2**2*z + z) +
                   (T**2*(m1 - m2) + T*m2*z*m1_sq_minus1 - m1*z**2*m2_sq_minus1**2)*atanhm2)*atanhm1*atanhm2 + 
                   (T**3*(-m1**2 + m2**2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*m2_sq_minus1*(T - m2**2*z + z)*atanhm2 + T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) - 
                    2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                   (T**3*(m1_sq_minus1-m2_sq_minus1)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) - 
                    2*T*(T**2*(m1 - m2) - T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 - 
                    2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2),
                   
                   -(2*T*m2*(T - m1**2*z + z)**2*atanhm1**2 + (T**3*(m1_sq_minus1 + m2_sq_minus1)/m1_sq_minus1 - T**2*z*(5*m2_sq_minus1 + m1_sq_minus1) + 2*T*m1*(T - m2**2*z + z)**2*atanhm2 + 
                    T*z**2*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm2 + (T**3*(m1_sq_minus1 + m2_sq_minus1)/m2_sq_minus1 - 
                     T**2*z*(5*m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*(5*m2_sq_minus1 + m1_sq_minus1) - 2*T*(m1 + m2)*(T**2 - 2*T*z*(m1*m2 - 1) + z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*atanhm2 - 
                    z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm1)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        # print("Γ_T_xx:", Γ_T_xx, "\nΓ_h_xx:", Γ_h_xx)

        return np.array([Γ_T_xx, Γ_h_xx])