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

class RMetric:
    def metric(self, p):
        raise NotImplementedError("This method should be implemented in subclasses.")

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
    
    def inv_metric(self, p):
        """
        Compute the inverse of the metric tensor at a given point p.

        Parameters:
        p (array-like): The point at which to compute the inverse metric tensor.

        Returns:
        numpy.ndarray: The inverse of the metric tensor at point p.
        """
        return np.linalg.inv(self.metric(p))
    
    def metric_det(self, p):
        """
        Compute the determinant of the metric tensor at a given point p.

        Parameters:
        p (array-like): The point at which to compute the metric tensor.

        Returns:
        float: The determinant of the metric tensor.
        """
        return np.linalg.det(self.metric(p))

    def geonorm(self, p, a):
        """
        Compute the norm of a vector a at point p using the metric tensor.

        Parameters:
        p (array-like): The point at which to compute the norm.
        a (array-like): The vector for which to compute the norm.

        Returns:
        float: The norm of the vector a at point p.
        """
        return np.sqrt(a.T @ self.metric(p) @ a)
        
    def geoip(self, p, a, b):
        """
        Compute the inner product of two vectors a and b at point p using the metric tensor.

        Parameters:
        p (array-like): The point at which to compute the inner product.
        a (array-like): The first vector.
        b (array-like): The second vector.

        Returns:
        float: The inner product of vectors a and b at point p.
        """
        return a.T @ self.metric(p) @ b

    def chrystoffel_func(self, p):
        """
        Compute the Christoffel symbols at a given point p.

        Parameters:
        p (array-like): The point at which to compute the Christoffel symbols.

        Returns:
        numpy.ndarray: The Christoffel symbols at point p.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
        
class Sphere(RMetric):
    dim = 2

    def metric(self, p):
        """
        Compute the metric tensor for a sphere at a given point.
        Parameters:
        -----------
        p : array-like
            A 2-dimensional vector of the form (theta, phi).
        Returns:
        --------
        numpy.ndarray
            A 2x2 diagonal matrix representing the metric tensor at the given point.
            The diagonal elements are equal to the square of the radius of the sphere,
            where the radius is the Euclidean norm of the input point `p`.
        """
        return np.diag([1, np.sin(p[0])**2])  

    def inv_metric(self, p):
        return np.diag([1, 1/np.sin(p[0])**2])
    
    def metric_det(self, p):
        return np.sin(p[0])**2

class AntiFerro(RMetric):
    dim = 2
    z = 1

    #                                        x=(T,h)
    def free_energy_non_minimized(self, m_s, x):
        T,h = x
        return 0.5*(self.z*(m_s[0]*m_s[1]) - h*(m_s[0] + m_s[1]) + 0.5 * T *(
        sp.special.xlogy(1+m_s[0], 1+m_s[0]) + sp.special.xlogy(1-m_s[0], 1-m_s[0]) +
        sp.special.xlogy(1+m_s[1], 1+m_s[1]) + sp.special.xlogy(1-m_s[1], 1-m_s[1]) )
        )

    #                       x=(T,h) 
    def tranceqn(self, m_s, x):
        m1 = np.tanh( (x[1] - self.z * m_s[1])/x[0] )
        m2 = np.tanh( (x[1] - self.z * m_s[0])/x[0] )
        return np.array([m1, m2])

    #                           x=(T,h) 
    def get_m_sublattices2(self, x, grid=500):
        assert np.nan not in x, "x contains NaN values"
        if self._phase_transition_line(x[0]) == np.abs(x[1]):
            return (np.sign(x[1]) * np.sqrt(1-x[0]/self.z) ,) * 2
        elif self._phase_transition_line(x[0]) < np.abs(x[1]) or x[0] > 1/self.z:
            # print(x)
            minim = sp.optimize.minimize(lambda m: np.abs(x[1]-x[0]*np.arctanh(m) - self.z*m), 0.5, bounds=((-1+6e-17,1-6e-17),), method="Powell")
            if minim.success and minim.fun < 1e-4:
                m = minim.x[0]
                return (m, m)
        elif np.abs(x[1]) >= 0.98*self._phase_transition_line(x[0]):
            mc = np.sign(x[1]) * np.sqrt(1-x[0]/self.z)
            hc = np.sign(x[1]) * self._phase_transition_line(x[0])
            m = mc + (1+3*mc**2)*(x[1]-hc)/2
            s = np.sqrt(3 * (1 - mc**2) / self.z * (- mc * (x[1]-hc)))
            if not np.abs(m+s)>1 and not np.abs(m-s)>1:
                return (m+s, m-s)
            elif (np.isclose(np.abs(m+s),1, rtol=0, atol=1e-4) and np.abs(m-s)<1):
                return (np.sign(m+s)*0.99999, m-s)
            elif (np.isclose(np.abs(m-s),1, rtol=0, atol=1e-4) and np.abs(m+s)<1):
                return (m+s, np.sign(m-s)*0.99999)
        elif x[0] >= 0.98/self.z:
            mc = np.sign(x[1]) * np.sqrt(1-x[0]/self.z) / 2
            Tc = (1-mc**2)*self.z
            hc = np.sign(x[1]) * self._phase_transition_line(Tc)
            m = mc + (1+3*mc**2)*(x[1]-hc)/2 + (3 *mc - (1 + 3 * mc**2)*np.arctanh(mc))*(x[0]-Tc)/(2 * self.z)
            s = np.sqrt(3 * (1 - mc**2) / self.z * ((mc*np.arctanh(mc) - 1) * (x[0] - Tc)- mc * (x[1]-hc)))
            if not np.abs(m+s)>1 and not np.abs(m-s)>1:
                return (m+s, m-s)
            elif (np.isclose(np.abs(m+s),1, rtol=0, atol=1e-4) and np.abs(m-s)<1):
                return (np.sign(m+s)*0.99999, m-s)
            elif (np.isclose(np.abs(m-s),1, rtol=0, atol=1e-4) and np.abs(m+s)<1):
                return (m+s, np.sign(m-s)*0.99999)
        
        # print(x)
        M1, M2 = np.meshgrid(*np.linspace([-1+1e-3,-1+1e-3],[1-1e-3,1-1e-3],grid).T)
        f = self.free_energy_non_minimized((M1,M2), x)
        ix, iy = np.unravel_index(np.argmin(f), f.shape)
        m1_0, m2_0 = M1[ix, iy], M2[ix,iy]
        if m1_0 == m2_0:
            m1_0 = np.min([m1_0 + 1e-2, 0.999])
            m2_0 = np.max([m2_0 - 1e-2, -0.999])
        # print("m0:", (m1_0, m2_0))
        m_s = fixed_point(self.tranceqn, (m1_0,m2_0), args=(x,))
        return m_s
    
    def get_m_sublattices(self, x, grid=30, tol=1e-16):
            assert np.nan not in x, "x contains NaN values"
            if self._phase_transition_line(x[0]) == np.abs(x[1]):
                return (np.sign(x[1]) * np.sqrt(1-x[0]/self.z) ,) * 2
            elif self._phase_transition_line(x[0]) < np.abs(x[1]) or x[0] > 1/self.z:
                # print(x)
                minim = sp.optimize.minimize(lambda m: np.abs(x[1]-x[0]*np.arctanh(m) - self.z*m), 0.5, bounds=((-1+6e-17,1-6e-17),), method="Powell")
                if minim.success and minim.fun < 1e-4:
                    m = minim.x[0]
                    return (m, m)
            elif np.abs(x[1]) >= 0.98*self._phase_transition_line(x[0]):
                mc = np.sign(x[1]) * np.sqrt(1-x[0]/self.z)
                hc = np.sign(x[1]) * self._phase_transition_line(x[0])
                m = mc + (1+3*mc**2)*(x[1]-hc)/2
                s = np.sqrt(3 * (1 - mc**2) / self.z * (- mc * (x[1]-hc)))
                if not np.abs(m+s)>1 and not np.abs(m-s)>1:
                    return (m+s, m-s)
                elif (np.isclose(np.abs(m+s),1, rtol=0, atol=1e-4) and np.abs(m-s)<1):
                    return (np.sign(m+s)*0.99999, m-s)
                elif (np.isclose(np.abs(m-s),1, rtol=0, atol=1e-4) and np.abs(m+s)<1):
                    return (m+s, np.sign(m-s)*0.99999)
            elif x[0] >= 0.98/self.z:
                mc = np.sign(x[1]) * np.sqrt(1-x[0]/self.z) / 2
                Tc = (1-mc**2)*self.z
                hc = np.sign(x[1]) * self._phase_transition_line(Tc)
                m = mc + (1+3*mc**2)*(x[1]-hc)/2 + (3 *mc - (1 + 3 * mc**2)*np.arctanh(mc))*(x[0]-Tc)/(2 * self.z)
                s = np.sqrt(3 * (1 - mc**2) / self.z * ((mc*np.arctanh(mc) - 1) * (x[0] - Tc)- mc * (x[1]-hc)))
                if not np.abs(m+s)>1 and not np.abs(m-s)>1:
                    return (m+s, m-s)
                elif (np.isclose(np.abs(m+s),1, rtol=0, atol=1e-4) and np.abs(m-s)<1):
                    return (np.sign(m+s)*0.99999, m-s)
                elif (np.isclose(np.abs(m-s),1, rtol=0, atol=1e-4) and np.abs(m+s)<1):
                    return (m+s, np.sign(m-s)*0.99999)
            
            def mins_on_grid(rng):
                M1, M2 = np.meshgrid(*np.linspace(*rng,grid).T)
                f = self.free_energy_non_minimized((M1,M2), x)
                ix, iy = np.unravel_index(np.argmin(f), f.shape)
                return (M1[ix, iy], M2[ix,iy])
            # print(x)
            delta = 1
            rng = np.array([[-1+1e-8,-1+1e-8],[1-1e-8,1-1e-8]])
            while delta > tol:
                m1, m2 = mins_on_grid(rng)
                delta = np.diff(rng, axis=0).max() / (grid - 1)
                rng = np.array([[max(m1-delta, -1+1e-8), max(m2-delta, -1+1e-8)],
                                [min(m1+delta, 1-1e-8), min(m2+delta, 1-1e-8)]])
            if m1 == m2:
                m1 = np.min([m1 + 1e-5, 0.999])
                m2 = np.max([m2 - 1e-5, -0.999])
            # print("m0:", (m1_0, m2_0))
            m_s = fixed_point(self.tranceqn, (m1,m2), args=(x,))
            return m_s

    #                x=(T,h)
    def metric(self, x):
        z = self.z
        T,h = x
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-9) and m1 < 1:
            one_minus_m1_sq = 1 - m1**2
            atanhm1 = np.arctanh(m1)
        else:
            one_minus_m1_sq = 0
            atanhm1 = 100 # large value to avoid divide by zero error
        
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-9) and m2 < 1:
            one_minus_m2_sq = 1 - m2**2
            atanhm2 = np.arctanh(m2)
        else:
            one_minus_m2_sq = 0
            atanhm2 = 100

        g_TT = (T*((one_minus_m1_sq)*atanhm1**2 + (one_minus_m2_sq)*atanhm2**2) -
            2*z*(one_minus_m1_sq)*(one_minus_m2_sq)*atanhm1*atanhm2)/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_Th = (-(T - z*one_minus_m1_sq)*one_minus_m2_sq*atanhm2 - (T - z*one_minus_m2_sq)*one_minus_m1_sq*atanhm1)/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_hh = (T*(-m1**2 - m2**2 + 2) - 2*z*(one_minus_m1_sq)*(one_minus_m2_sq))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        return np.array([[g_TT, g_Th], [g_Th, g_hh]])

    def inv_metric(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-16):
            m1_sq_minus1 = m1**2 - 1
            atanhm1 = np.arctanh(m1)
        else:
            m1_sq_minus1 = 1e-16
            atanhm1 = 100 # large value to avoid divide by zero error
        
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-16):
            m2_sq_minus1 = m2**2 - 1
            atanhm2 = np.arctanh(m2)
        else:
            m2_sq_minus1 = 1e-16
            atanhm2 = -100

            
        sec_diag = -(2*(T + z*m1_sq_minus1)*m2_sq_minus1*atanhm2 + 2*(T + z*m2_sq_minus1)*m1_sq_minus1*atanhm1)/(m1_sq_minus1*m2_sq_minus1*(atanhm1 - atanhm2)**2)
        return np.array([[(-2*T*(m1_sq_minus1+m2_sq_minus1)/(m1_sq_minus1*m2_sq_minus1) - 4*z)/(atanhm1 - atanhm2)**2, sec_diag],
            [sec_diag, (-2*T*atanhm1**2/m2_sq_minus1 - 2*T*atanhm2**2/m1_sq_minus1 - 4*z*atanhm1*atanhm2)/(atanhm1 - atanhm2)**2]])

    def metric_det(self, x):
        """
        Compute the determinant of the metric tensor at a given point x.

        Parameters:
        x (array-like): The point at which to compute the metric tensor.

        Returns:
        float: The determinant of the metric tensor.
        """
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-16) and m1 < 1:
            m1_sq_minus1 = m1**2 - 1
            atanhm1 = np.arctanh(m1)
        else:
            m1_sq_minus1 = 1e-16
            atanhm1 = 100 # large value to avoid divide by zero error
        
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-16) and m2 < 1:
            m2_sq_minus1 = m2**2 - 1
            atanhm2 = np.arctanh(m2)
        else:
            m2_sq_minus1 = 1e-16
            atanhm2 = -100
        
        return m1_sq_minus1*m2_sq_minus1*(atanhm1 - atanhm2)**2/(4*T**2 - 4*z**2*m1_sq_minus1*m2_sq_minus1)

    def christoffel_func(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        
        if not np.isclose(m1, 1, rtol=0, atol=1e-9) and m1 < 1:
            m1_sq_minus1 = m1**2 - 1
            atanhm1 = np.arctanh(m1)
        else:
            m1_sq_minus1 = 0
            atanhm1 = 100 # large value to avoid divide by zero error
        
        if not np.isclose(m2, 1, rtol=0, atol=1e-9) and m2 < 1:
            m2_sq_minus1 = m2**2 - 1
            atanhm2 = np.arctanh(m2)
        else:
            m2_sq_minus1 = 0
            atanhm2 = 100

        Γ_T_xx = [[(2*T*(T**2*m1 - m2*z**2*m1_sq_minus1**2)*atanhm1**3 + (-T**3*(3*m1**2 + m2**2 - 4)/m1_sq_minus1 + T*z**2*(m1 - m2)*(m1 + m2)*m2_sq_minus1 + 
            2*T*(T**2*m2 - m1*z**2*m2_sq_minus1**2)*atanhm2 + 4*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + (-T**3*(m1**2 + 3*m2**2 - 4)/m2_sq_minus1 - 
            T*z**2*(m1 - m2)*(m1 + m2)*m1_sq_minus1 + 2*T*(-T**2*m1 + 2*T*z*(m1 - m2)*(m1*m2 + 1) + m2*z**2*m1_sq_minus1**2)*atanhm2 + 4*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + 
            2*(2*T**3 + 3*T**2*z*(m1**2 + m2**2 - 2) - 6*T*z**2*m1_sq_minus1*m2_sq_minus1 + T*(-T**2*m2 + 2*T*z*(-m1 + m2)*(m1*m2 + 1) + m1*z**2*m2_sq_minus1**2)*atanhm2 - 
            z**3*m1_sq_minus1*m2_sq_minus1*(m1**2 + m2**2 - 2))*atanhm1*atanhm2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2), 
            
            (-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - 
            (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) - 3*T**2*z*m1_sq_minus1*(m1**2 + m2**2 - 2) + 
            T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1**2 + m2**2 - 6) - 2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - 
            m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
            (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) - 3*T**2*z*m2_sq_minus1*(m1**2 + m2**2 - 2) + T*z**2*m1_sq_minus1*m2_sq_minus1*(m1**2 + 5*m2**2 - 6) +
            2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/
            (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2)], 
            
            [(-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - 
            (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) - 3*T**2*z*m1_sq_minus1*(m1**2 + m2**2 - 2) + 
            T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1**2 + m2**2 - 6) - 2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - 
            m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
            (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) - 3*T**2*z*m2_sq_minus1*(m1**2 + m2**2 - 2) + 
            T*z**2*m1_sq_minus1*m2_sq_minus1*(m1**2 + 5*m2**2 - 6) + 2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - 
            z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),

            -(T**3*(m1**2 + m2**2 - 2)**2/(m1_sq_minus1*m2_sq_minus1) - 6*T**2*z*(m1**2 + m2**2 - 2) + T*z**2*(m1**4 + 2*m1**2*(5*m2**2 - 6) + m2**4 - 12*m2**2 + 12) + 
            2*T*(m1 - m2)*(-atanhm1 + atanhm2)*(T**2 + 2*T*z*(m1*m2 + 1) - m1*m2*z**2*(m1**2 + m1*m2 + m2**2 - 2) + z**2) - 2*z**3*m1_sq_minus1*m2_sq_minus1*(m1**2 + m2**2 - 2))/
            (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        
        Γ_h_xx = [[(-2*T*m2*z**2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*m1_sq_minus1**2*atanhm1**4 +
            (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*m2_sq_minus1*(T**3 - T**2*z*m1_sq_minus1 + 2*T*m1*z**2*m1_sq_minus1*m2_sq_minus1*atanhm2 + T*z**2*m1_sq_minus1*m2_sq_minus1 +
            3*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm2**3/m1_sq_minus1 + (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-T**3 - 7*T**2*z*m2_sq_minus1 - 5*T*z**2*m1_sq_minus1*m2_sq_minus1 + 
            2*T*(T**2*m2 - 2*T*m1*z*m2_sq_minus1 + m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm1*atanhm2**2 - 
            (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3 + 7*T**2*z*m1_sq_minus1 + 2*T**2*(m1 + m2)*(T - 2*m1*m2*z + 2*z)*atanhm2 + 5*T*z**2*m1_sq_minus1*m2_sq_minus1 -
            z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2*atanhm2 - (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-2*T*m2_sq_minus1*(T**2*m1 - 2*T*m2*z*m1_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 +
            m1_sq_minus1*(T**3 - T**2*z*m2_sq_minus1 + T*z**2*m1_sq_minus1*m2_sq_minus1 + 3*z**3*m1_sq_minus1*m2_sq_minus1**2))*atanhm1**3/m2_sq_minus1)/
            (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),
            
            -(-2*T*m2*z*(T + z*m1_sq_minus1)*m1_sq_minus1*atanhm1**3 + 2*T*(-2*(T + z*m1_sq_minus1)*(T + z*m2_sq_minus1) + 
            (T**2*(-m1 + m2) + T*m2*z*m1_sq_minus1 + m1*z**2*m2_sq_minus1**2)*atanhm2)*atanhm1*atanhm2 + 
            (T**3*(-m1**2 + m2**2)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 - T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) + 2*T*(T**2*(m1 - m2) + T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 -
            2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + (T**3*(m1 - m2)*(m1 + m2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*(T + z*m2_sq_minus1)*m2_sq_minus1*atanhm2 -
            T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) - 2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)], 
            
            [-(-2*T*m2*z*(T + z*m1_sq_minus1)*m1_sq_minus1*atanhm1**3 + 2*T*(-2*(T + z*m1_sq_minus1)*(T + z*m2_sq_minus1) +
            (T**2*(-m1 + m2) + T*m2*z*m1_sq_minus1 + m1*z**2*m2_sq_minus1**2)*atanhm2)*atanhm1*atanhm2 + (T**3*(-m1**2 + m2**2)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 -
            T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) + 2*T*(T**2*(m1 - m2) + T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 - 2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 +
            (T**3*(m1 - m2)*(m1 + m2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*(T + z*m2_sq_minus1)*m2_sq_minus1*atanhm2 - T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) -
            2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2),
            
            (-2*T*m2*(T + z*m1_sq_minus1)**2*atanhm1**2 +(-T**3*(m1**2 + m2**2 - 2)/m1_sq_minus1 - T**2*z*(m1**2 + 5*m2**2 - 6) - 
            2*T*m1*(T + z*m2_sq_minus1)**2*atanhm2 - T*z**2*m2_sq_minus1*(5*m1**2 + m2**2 - 6) -
            z**3*m1_sq_minus1*m2_sq_minus1*(m1**2 + m2**2 - 2))*atanhm2 + (-T**3*(m1**2 + m2**2 - 2)/m2_sq_minus1 - T**2*z*(5*m1**2 + m2**2 - 6) - T*z**2*m1_sq_minus1*(m1**2 + 5*m2**2 - 6) +
            2*T*(m1 + m2)*(T**2 + 2*T*z*(m1*m2 - 1) + z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1*(m1**2 + m2**2 - 2))*atanhm1)/
            (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        # print("Γ_T_xx:", Γ_T_xx, "\nΓ_h_xx:", Γ_h_xx)

        return np.array([Γ_T_xx, Γ_h_xx])

    def _phase_transition_line(self, T):
        """
        Compute the phase transition line for a given temperature T.

        Parameters:
        T (float): The temperature at which to compute the phase transition line.

        Returns:
        float: The value of h at the phase transition line.
        """
        return T/2 * np.log((1+np.sqrt(1-T/self.z))/(1-np.sqrt(1-T/self.z))) + self.z * np.sqrt(1-T/self.z) if T <= self.z else np.nan

    def phase_transition_line(self, T):
        """
        Compute the phase transition line for a given temperature T.

        Parameters:
        T (float): The temperature at which to compute the phase transition line.

        Returns:
        float: The value of h at the phase transition line.
        """
        return self._phase_transition_line(T)

    def is_ordered_phase(self, x):
        """
        Check if the system is in the ordered phase for a given set of parameters x.

        Parameters:
        x (array-like): The parameters (T, h) to check.

        Returns:
        bool: True if the system is in the ordered (Anti-Ferromagnetic) phase, False otherwise.
        """
        T, h = x
        if T > 1/self.z:
            return False
        hc = self._phase_transition_line(T)
        return (h < hc) and (h > -hc)
    
class AntiFerroSivak(AntiFerro):
    Γ = 1

    def get_m_sublattices(self, x, grid=500):
        (β, βh) = x
        return super().get_m_sublattices((1/β, βh/β), grid)

    def metric(self, x):
        z = self.z
        Γ = self.Γ
        β, βh = x
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-9) and m1 < 1:
            ζ1 = 1 - m1**2
        else:
            ζ1 = 1e-18
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-9) and m2 < 1:
            ζ2 = 1 - m2**2
        else:
            ζ2 = 1e-19
        
        return β/ (Γ * 2 * (1 - β**2 * z**2 * ζ1 * ζ2)**2) * np.array([
            [z ** 2 * ((1 + β**2 * z**2 * ζ1 * ζ2) * (m1**2 * ζ2 + m2**2 * ζ1) - 4 * β * z * m1 * m2 * ζ1 * ζ2),
             z * ((1 + β**2 * z**2 * ζ1 * ζ2) * (m1 * ζ2 + m2 * ζ1) - 2 * β * z * (m1 + m2) * ζ1 * ζ2)],
            [z * ((1 + β**2 * z**2 * ζ1 * ζ2) * (m1 * ζ2 + m2 * ζ1) - 2 * β * z * (m1 + m2) * ζ1 * ζ2),
             (1 + β**2 * z**2 * ζ1 * ζ2) * (ζ1 + ζ2) - 4 * β * z * ζ1 * ζ2]
        ])

    def inv_metric(self, x):
        z = self.z
        Γ = self.Γ
        β, βh = x
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-9) and m1 < 1:
            ζ1 = 1 - m1**2
        else:
            ζ1 = 0
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-9) and m2 < 1:
            ζ2 = 1 - m2**2
        else:
            ζ2 = 0
        if np.isclose(m1, m2, rtol=0, atol=1e-16):
            m2 += 1e-16
            print("Warning: m1 and m2 are too close, adjusting m2 slightly to avoid singularity in inverse metric calculation.")
        
        return 2*Γ/ ((m1-m2)**2 * β * ζ1 * ζ2 * z**2) * np.array([
            [(ζ1 + ζ2 - 4 * β * z * ζ1 * ζ2 + z**2 * β**2 * ζ1 * ζ2 * (ζ1 + ζ2)) / z**2,
             -(m1 * ζ2 *(1-2*z*β * ζ1 + z**2 * β**2 * ζ1 * ζ2) + m2 * ζ1 *(1-2*z*β * ζ2 + z**2 * β**2 * ζ1 * ζ2)) / z],
            [-(m1 * ζ2 *(1-2*z*β * ζ1 + z**2 * β**2 * ζ1 * ζ2) + m2 * ζ1 *(1-2*z*β * ζ2 + z**2 * β**2 * ζ1 * ζ2)) / z,
             -4 * z * m1 * m2 * β * ζ1 * (ζ2 + m2**2 + m1**2 * ζ2) * ζ1 *(1+z**2 * β**2 * ζ1 * ζ2)
        ]])
    
    def metric_det(self, x):
        z = self.z
        Γ = self.Γ
        β, βh = x
        m1, m2 = self.get_m_sublattices(x)

        if not np.isclose(np.abs(m1), 1, rtol=0, atol=1e-9) and m1 < 1:
            ζ1 = 1 - m1**2
        else:
            ζ1 = 1e-18
        if not np.isclose(np.abs(m2), 1, rtol=0, atol=1e-9) and m2 < 1:
            ζ2 = 1 - m2**2
        else:
            ζ2 = 1e-18

        return (z**2 * β ** 2 * ζ1 * ζ2 * (m1-m2)**2) / (Γ**2 * 4 * (1 - β**2 * z**2 * ζ1 * ζ2)**2)

    def christoffel_func(self, x):
        beta, _ = x
        mA, mB = self.get_m_sublattices(x)

        if not np.isclose(np.abs(mA), 1, rtol=0, atol=1e-9) and mA < 1:
            zetaA = 1 - mA**2
        else:
            zetaA = 1e-18
        if not np.isclose(np.abs(mB), 1, rtol=0, atol=1e-9) and mB < 1:
            zetaB = 1 - mB**2
        else:
            zetaB = 1e-19

        Γ1_11 = ((((0.5 * ((mA -mB) ** -2) * ((-1 + (beta ** 2) * zetaA * zetaB) ** 
        -3) * (2 * (mA ** 6) * (beta ** 3) * (zetaB ** 2) * (8 + 3 * zetaB + 
        (beta ** 5) * (zetaA ** 2) * (zetaA -2 * zetaB) * (zetaB ** 2) + 
        (beta ** 4) * (zetaA ** 2) * (zetaB ** 2) * (10 + zetaB) + 2 * (beta ** 
        2) * zetaA * zetaB * (11 + 2 * zetaB) -2 * (beta ** 3) * zetaA * 
        zetaB * (4 * zetaB + zetaA * (6 + zetaB)) -(beta * (6 * zetaB + zetaA 
        * (13 + 6 * zetaB))) + (mB ** 2) * (-8 -22 * (beta ** 2) * zetaA * 
        zetaB -10 * (beta ** 4) * (zetaA ** 2) * (zetaB ** 2) -((beta ** 5) * 
        (zetaA ** 2) * (zetaA -2 * zetaB) * (zetaB ** 2)) + 4 * (beta ** 3) * 
        zetaA * zetaB * (3 * zetaA + 2 * zetaB) + beta * (13 * zetaA + 6 * 
        zetaB))) + 2 * (mA ** 5) * mB * (beta ** 2) * zetaB * (-3 * (beta ** 
        6) * (zetaA ** 3) * (zetaB ** 3) -3 * (1 + 4 * zetaB) -((beta ** 4) * 
        (zetaA ** 2) * (zetaB ** 2) * (29 + 14 * zetaB)) -((beta ** 2) * 
        zetaA * zetaB * (29 + 30 * zetaB)) + (beta ** 5) * (zetaA ** 2) * 
        (zetaB ** 2) * (4 * zetaA + zetaB * (6 + zetaB)) + (mB ** 2) * (3 -10 
        * beta * zetaA + 29 * (beta ** 2) * zetaA * zetaB + 29 * (beta ** 4) 
        * (zetaA ** 2) * (zetaB ** 2) + 3 * (beta ** 6) * (zetaA ** 3) * 
        (zetaB ** 3) -2 * (beta ** 5) * (zetaA ** 2) * (zetaB ** 2) * (2 * 
        zetaA + 3 * zetaB) -2 * (beta ** 3) * zetaA * zetaB * (13 * zetaA + 9 
        * zetaB)) + beta * (3 * (zetaB ** 2) + 2 * zetaA * (5 + 12 * zetaB)) 
        + 2 * (beta ** 3) * zetaA * zetaB * (zetaB * (9 + 2 * zetaB) + zetaA 
        * (13 + 12 * zetaB))) + 2 * (mA ** 3) * mB * beta * (zetaB * (-2 + 
        (mB ** 2) * (1 -3 * beta) -3 * (beta ** 2) * (zetaB ** 2) + 3 * beta 
        * (1 + 4 * zetaB)) + (-1 + mB ** 2) * (beta ** 6) * (zetaA ** 4) * 
        (zetaB ** 2) * (6 * zetaB * (-2 + beta * zetaB) + (mB ** 2) * (-14 + 
        3 * beta * zetaB)) + (beta ** 4) * (zetaA ** 3) * zetaB * ((mB ** 4) 
        * (-34 + 29 * beta * zetaB + 4 * (beta ** 2) * (zetaB ** 2)) + zetaB 
        * (7 -2 * (-6 + 5 * beta + 2 * (beta ** 2)) * zetaB + 3 * beta * (-2 
        + beta ** 2) * (zetaB ** 2)) -((mB ** 2) * (-34 + zetaB + 29 * beta * 
        zetaB -10 * beta * (zetaB ** 2) + 3 * (beta ** 3) * (zetaB ** 3)))) 
        -((beta ** 2) * (zetaA ** 2) * ((mB ** 4) * (8 -29 * beta * zetaB + 
        10 * (beta ** 2) * (zetaB ** 2)) + (mB ** 2) * (-8 + (-20 + 29 * 
        beta) * zetaB + 18 * (1 -2 * beta) * beta * (zetaB ** 2) + (beta ** 
        2) * (1 + 29 * beta) * (zetaB ** 3) -6 * (beta ** 4) * (zetaB ** 4)) 
        + zetaB * (12 + 2 * (6 -6 * beta + 13 * (beta ** 2)) * zetaB + (26 
        -29 * beta) * (beta ** 2) * (zetaB ** 2) + 2 * (beta ** 3) * (-7 + 3 
        * beta) * (zetaB ** 3) + (beta ** 4) * (zetaB ** 4)))) + zetaA * (1 + 
        6 * beta * zetaB * (2 + zetaB) -2 * (beta ** 4) * (zetaB ** 3) * (9 + 
        2 * zetaB) -2 * (beta ** 2) * zetaB * (5 + 18 * zetaB) + (beta ** 3) 
        * (zetaB ** 2) * (29 + 30 * zetaB) + (mB ** 4) * beta * (3 -2 * beta 
        * zetaB) + (mB ** 2) * (1 -29 * (beta ** 3) * (zetaB ** 2) + 18 * 
        (beta ** 4) * (zetaB ** 3) + 4 * (beta ** 2) * zetaB * (3 + 2 * 
        zetaB) -(beta * (3 + 14 * zetaB))))) + (mA ** 4) * beta * (2 * (mB ** 
        4) * beta * (zetaB + (beta ** 4) * (zetaA ** 3) * (zetaB ** 2) * (27 
        -16 * beta * zetaB) + (beta ** 2) * (zetaA ** 2) * zetaB * (28 -44 * 
        beta * zetaB + 11 * (beta ** 2) * (zetaB ** 2)) + zetaA * (1 -20 * 
        beta * zetaB + 12 * (beta ** 2) * (zetaB ** 2))) + zetaB * (-2 * 
        zetaB -2 * (beta ** 2) * zetaB * (8 + zetaA * (-2 + zetaB) + 3 * 
        zetaB) + 2 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 3) * (-10 -zetaB 
        + zetaA * (14 + zetaB)) + 2 * (beta ** 4) * zetaA * (zetaB ** 2) * 
        (-22 -4 * zetaB + zetaA * (16 + zetaB)) + (beta ** 7) * (zetaA ** 2) 
        * (zetaB ** 3) * (3 * (zetaA ** 2) + 4 * zetaB -(zetaA * (2 + 3 * 
        zetaB))) + beta * (zetaB + zetaA * (-1 + 4 * zetaB)) + (beta ** 5) * 
        zetaA * (zetaB ** 2) * (zetaA * (24 -5 * zetaB) + 16 * zetaB -((zetaA 
        ** 2) * (47 + 4 * zetaB))) + (beta ** 3) * zetaB * (-3 * (zetaA ** 2) 
        + 12 * zetaB + zetaA * (26 + 7 * zetaB))) + (mB ** 2) * (6 * zetaB + 
        (beta ** 7) * (zetaA ** 2) * (zetaB ** 4) * (-3 * (zetaA ** 2) -4 * 
        zetaB + zetaA * (2 + 3 * zetaB)) -4 * (beta ** 6) * (zetaA ** 2) * 
        (zetaB ** 3) * (-5 * zetaB + zetaA * (-8 + 6 * zetaB)) + 8 * (beta ** 
        2) * zetaB * (2 * zetaB + zetaA * (5 + 6 * zetaB)) + 2 * (beta ** 4) 
        * zetaA * (zetaB ** 2) * (22 * zetaB + zetaA * (44 + 9 * zetaB)) 
        -(beta * ((2 -7 * zetaB) * zetaB + zetaA * (2 + 27 * zetaB))) + (beta 
        ** 5) * zetaA * (zetaB ** 2) * (-16 * (zetaB ** 2) + zetaA * zetaB * 
        (-46 + 9 * zetaB) + (zetaA ** 2) * (-54 + 31 * zetaB)) -((beta ** 3) 
        * zetaB * (12 * (zetaB ** 2) + zetaA * zetaB * (50 + 3 * zetaB) + 
        (zetaA ** 2) * (56 + 65 * zetaB))))) + (mA ** 2) * (2 * (mB ** 6) * 
        (beta ** 4) * (zetaA ** 2) * (-3 * zetaB + 2 * (beta ** 4) * (zetaA ** 
        3) * (zetaB ** 2) + zetaA * (6 -6 * beta * zetaB -4 * (beta ** 2) * 
        (zetaB ** 2)) -((beta ** 2) * (zetaA ** 2) * zetaB * (-8 + 2 * beta * 
        zetaB + (beta ** 2) * (zetaB ** 2)))) + (mB ** 2) * beta * ((-6 + 
        beta * (2 -7 * zetaB)) * zetaB -3 * (beta ** 7) * (zetaA ** 5) * 
        (zetaB ** 3) + (beta ** 5) * (zetaA ** 4) * (zetaB ** 2) * (7 -8 * 
        zetaB + 2 * beta * (2 + 3 * beta) * (zetaB ** 2)) + beta * (zetaA ** 
        2) * (9 + 8 * (1 -7 * beta + 7 * (beta ** 2)) * zetaB -2 * beta * (2 
        -57 * beta + 44 * (beta ** 2)) * (zetaB ** 2) + 2 * (beta ** 3) * (-9 
        + 11 * beta) * (zetaB ** 3) -9 * (beta ** 4) * (zetaB ** 4)) + (beta ** 
        3) * (zetaA ** 3) * zetaB * (35 -3 * (beta ** 4) * (zetaB ** 4) + 2 * 
        beta * zetaB * (-25 + 2 * zetaB) + 8 * (beta ** 3) * (zetaB ** 2) * 
        (-4 + 3 * zetaB) -6 * (beta ** 2) * zetaB * (-9 + 5 * zetaB)) + zetaA 
        * (-6 -4 * zetaB + 3 * (beta ** 3) * (zetaB ** 2) * (8 + zetaB) -8 * 
        (beta ** 2) * zetaB * (5 + 6 * zetaB) + beta * (2 + 38 * zetaB))) + 
        (mB ** 4) * beta * (-2 * beta * zetaB + (beta ** 7) * (zetaA ** 5) * 
        (zetaB ** 2) * (-4 + 3 * zetaB) + (beta ** 5) * (zetaA ** 4) * zetaB 
        * (-16 + (-7 + 4 * beta) * zetaB + 2 * (beta ** 2) * (zetaB ** 2) -3 
        * (beta ** 2) * (zetaB ** 3)) + (beta ** 3) * (zetaA ** 3) * (-12 + 
        (-35 + 12 * beta) * zetaB + 2 * (25 -23 * beta) * beta * (zetaB ** 2) 
        + (beta ** 2) * (-1 + 32 * beta) * (zetaB ** 3)) + zetaA * (6 + 40 * 
        (beta ** 2) * zetaB -24 * (beta ** 3) * (zetaB ** 2) -(beta * (2 + 11 
        * zetaB))) -(beta * (zetaA ** 2) * (9 -56 * beta * zetaB -88 * (beta ** 
        3) * (zetaB ** 2) + 22 * (beta ** 4) * (zetaB ** 3) + (beta ** 2) * 
        zetaB * (50 + 49 * zetaB)))) + zetaB * (-(((-1 + beta) ** 2) * zetaB) 
        + (beta ** 6) * (1 -2 * beta -3 * (beta ** 2)) * (zetaA ** 4) * 
        (zetaB ** 3) + (beta ** 2) * (zetaA ** 2) * zetaB * (-25 + 9 * (beta ** 
        4) * (zetaB ** 2) -2 * (beta ** 3) * zetaB * (16 + zetaB) + beta * 
        (-6 + 32 * zetaB) + (beta ** 2) * (3 -7 * (zetaB ** 2))) + zetaA * (1 
        + 2 * (beta ** 3) * (-2 + zetaB) * zetaB + 5 * (beta ** 4) * (zetaB ** 
        2) + 2 * beta * (1 + 8 * zetaB) + (beta ** 2) * (1 -4 * zetaB -7 * 
        (zetaB ** 2))) + (beta ** 4) * (zetaA ** 3) * (zetaB ** 2) * (-25 + 3 
        * (beta ** 4) * (zetaB ** 2) -2 * (beta ** 3) * zetaB * (14 + zetaB) 
        + 2 * beta * (3 + 8 * zetaB) + (beta ** 2) * (47 + 4 * zetaB -(zetaB ** 
        2))))) -2 * mA * ((mB ** 5) * (beta ** 2) * zetaA * (3 -2 * beta * 
        zetaB -((beta ** 5) * (zetaA ** 4) * (zetaB ** 2)) + (beta ** 3) * 
        (zetaA ** 3) * zetaB * (-4 + 6 * beta * zetaB -14 * (beta ** 2) * 
        zetaB + 3 * (beta ** 3) * (zetaB ** 2)) + zetaA * (4 + 29 * (beta ** 
        2) * zetaB -10 * (beta ** 3) * (zetaB ** 2) -8 * beta * (1 + zetaB)) 
        + beta * (zetaA ** 2) * (-3 + 14 * beta * zetaB + 29 * (beta ** 3) * 
        (zetaB ** 2) + 4 * (beta ** 4) * (zetaB ** 3) -2 * (beta ** 2) * 
        zetaB * (17 + 4 * zetaB))) -2 * mB * zetaA * zetaB * (-1 + beta * (3 
        * zetaA -zetaB) -6 * (beta ** 7) * (zetaA ** 3) * (zetaB ** 2) + 3 * 
        (beta ** 8) * (zetaA ** 3) * (zetaB ** 3) + (beta ** 4) * zetaA * 
        zetaB * (5 -7 * zetaA * zetaB) + (beta ** 6) * (zetaA ** 2) * (zetaB ** 
        2) * (9 + zetaA + 3 * zetaB -(zetaA * zetaB)) -((beta ** 2) * (1 + 
        zetaA + 3 * zetaB + 7 * zetaA * zetaB)) + 2 * (beta ** 3) * zetaB * 
        (1 + 5 * (zetaA ** 2) + zetaA * (4 + zetaB)) + (beta ** 5) * zetaA * 
        zetaB * (-2 * zetaB + 3 * (zetaA ** 2) * zetaB -(zetaA * (10 + 8 * 
        zetaB + zetaB ** 2)))) + (mB ** 3) * beta * (-zetaB + (beta ** 6) * 
        (zetaA ** 5) * (zetaB ** 2) + zetaA * (2 + 2 * (beta ** 2) * zetaB 
        -(beta * (3 + 4 * zetaB))) + (beta ** 4) * (zetaA ** 4) * zetaB * (4 
        + 2 * beta * (-3 + zetaB) * zetaB + 3 * (beta ** 3) * (zetaB ** 2) * 
        (-1 + 2 * zetaB) -2 * (beta ** 2) * zetaB * (-7 + 6 * zetaB)) + beta 
        * (zetaA ** 2) * ((beta ** 3) * (10 -7 * zetaB) * (zetaB ** 2) -2 * 
        (2 + zetaB) + (beta ** 2) * zetaB * (-29 + 4 * zetaB) + 4 * beta * (2 
        + 4 * zetaB + zetaB ** 2)) -((beta ** 2) * (zetaA ** 3) * (-3 + 14 * 
        beta * zetaB + (beta ** 3) * (29 -18 * zetaB) * (zetaB ** 2) + 4 * 
        (beta ** 4) * (zetaB ** 3) + 2 * (beta ** 2) * zetaB * (-17 + 3 * 
        zetaB + 2 * (zetaB ** 2)))))) + (mB ** 2) * zetaA * ((1 + 2 * beta 
        -((-1 + mB ** 2) * (beta ** 2))) * zetaB -((beta ** 6) * (zetaA ** 4) 
        * (zetaB ** 2) * (2 * (mB ** 4) * beta * (-1 + 2 * beta) + zetaB + 2 
        * beta * zetaB -3 * (beta ** 2) * zetaB + (mB ** 2) * beta * (2 -4 * 
        beta -2 * zetaB + 3 * beta * zetaB))) + zetaA * (-1 -2 * (-1 + mB ** 
        2) * beta + 2 * (beta ** 3) * zetaB * (2 -2 * (mB ** 2) + 5 * zetaB) 
        + (-1 + mB ** 2) * (beta ** 4) * zetaB * (6 * (mB ** 2) + 13 * zetaB) 
        + (beta ** 2) * (-1 + mB ** 2 -12 * zetaB + 4 * (mB ** 2) * zetaB + 7 
        * (zetaB ** 2))) + (beta ** 4) * (zetaA ** 3) * zetaB * (2 * (mB ** 
        4) * beta * (4 + 2 * (beta ** 2) * zetaB + (beta ** 3) * (zetaB ** 2) 
        -2 * beta * (4 + zetaB)) + (mB ** 2) * beta * (2 * (-4 + zetaB) + 4 * 
        (beta ** 2) * (-1 + zetaB) * zetaB + (beta ** 3) * (zetaB ** 2) * (-2 
        + 3 * zetaB) + beta * (16 -5 * zetaB -4 * (zetaB ** 2))) + zetaB * 
        (-7 -2 * beta -3 * (beta ** 4) * (zetaB ** 2) -2 * (beta ** 3) * 
        zetaB * (2 + zetaB) + (beta ** 2) * (9 + 12 * zetaB + zetaB ** 2))) + 
        (beta ** 2) * (zetaA ** 2) * (2 * (mB ** 4) * beta * (3 + 6 * (beta ** 
        2) * zetaB + 4 * (beta ** 3) * (zetaB ** 2) -6 * beta * (1 + zetaB)) 
        + (mB ** 2) * beta * (-12 * (beta ** 2) * zetaB + (beta ** 3) * (-8 + 
        zetaB) * (zetaB ** 2) -2 * (3 + zetaB) + beta * (12 + 7 * zetaB)) + 
        zetaB * (-7 + 2 * beta -10 * (beta ** 3) * (zetaB ** 2) -((beta ** 4) 
        * (zetaB ** 2)) + (beta ** 2) * (5 + 7 * (zetaB ** 2))))))) / zetaB) 
        / zetaA) / (-1 + (-1 + mA ** 2) * (-1 + mB ** 2) * (beta ** 2))) / beta
        Γ1_12 = (((0.5 * ((mA -mB) ** -2) * ((-1 + (beta ** 2) * zetaA * zetaB) ** 
        -3) * (((mA * zetaB * (1 -2 * beta * zetaA + (beta ** 2) * zetaA * 
        zetaB) + mB * zetaA * (1 -2 * beta * zetaB + (beta ** 2) * zetaA * 
        zetaB)) * (-2 * (mB ** 4) * (beta ** 2) * (-1 + beta * zetaA) * (-1 + 
        3 * beta * zetaA -3 * (beta ** 2) * zetaA * zetaB + (beta ** 3) * 
        (zetaA ** 2) * zetaB) + 2 * (mA ** 3) * mB * beta * (-1 + beta * 
        zetaB) * (-1 + 3 * beta * zetaB -3 * (beta ** 2) * zetaA * zetaB + 
        (beta ** 3) * zetaA * (zetaB ** 2)) + 2 * (mA ** 4) * (-1 + mB ** 2) 
        * (beta ** 2) * (-1 + beta * zetaB) * (-1 + 3 * beta * zetaB -3 * 
        (beta ** 2) * zetaA * zetaB + (beta ** 3) * zetaA * (zetaB ** 2)) + 2 
        * mA * mB * beta * (-2 + 4 * beta * (zetaA + zetaB) + 4 * (beta ** 3) 
        * zetaA * zetaB * (zetaA + zetaB) -3 * (beta ** 2) * ((zetaA + zetaB) 
        ** 2) + (mB ** 2) * (-1 + beta * zetaA) * (-1 + 3 * beta * zetaA -3 * 
        (beta ** 2) * zetaA * zetaB + (beta ** 3) * (zetaA ** 2) * zetaB) 
        -((beta ** 4) * zetaA * zetaB * (zetaA ** 2 + zetaB ** 2))) + (-1 + 
        beta ** 2) * (zetaB + (beta ** 4) * (zetaA ** 3) * (zetaB ** 2) + 
        (beta ** 2) * (zetaA ** 2) * zetaB * (6 -8 * beta * zetaB + (beta ** 
        2) * (zetaB ** 2)) + zetaA * (1 -8 * beta * zetaB + 6 * (beta ** 2) * 
        (zetaB ** 2))) -((mB ** 2) * (beta ** 2) * (-2 + zetaB + (beta ** 4) 
        * (zetaA ** 3) * (-2 + zetaB) * zetaB + zetaA * (1 -8 * beta * (-1 + 
        zetaB) + 6 * (beta ** 2) * (-1 + zetaB) * zetaB) + (beta ** 2) * 
        (zetaA ** 2) * (-6 + (6 + 8 * beta) * zetaB -8 * beta * (zetaB ** 2) 
        + (beta ** 2) * (zetaB ** 3)))) + (mA ** 2) * (-1 + mB ** 2) * (beta ** 
        2) * (-2 + zetaB + 8 * beta * zetaB -6 * (beta ** 2) * (zetaB ** 2) + 
        (beta ** 4) * (zetaA ** 3) * (zetaB ** 2) + 2 * (mB ** 2) * (-1 + 
        beta * zetaA) * (-1 + 3 * beta * zetaA -3 * (beta ** 2) * zetaA * 
        zetaB + (beta ** 3) * (zetaA ** 2) * zetaB) + (beta ** 2) * (zetaA ** 
        2) * zetaB * (6 -8 * beta * zetaB + (beta ** 2) * (zetaB ** 2)) + 
        zetaA * (1 -8 * beta * zetaB + 6 * (beta ** 2) * (-1 + zetaB) * zetaB 
        + 8 * (beta ** 3) * (zetaB ** 2) -2 * (beta ** 4) * (zetaB ** 3))))) 
        / beta + 2 * (zetaA + zetaB -4 * beta * zetaA * zetaB + (beta ** 2) * 
        (zetaA ** 2) * zetaB + (beta ** 2) * zetaA * (zetaB ** 2)) * ((mA ** 
        5) * (beta ** 2) * (1 + (-1 + mB ** 2) * beta) * (zetaB ** 2) * (3 + 
        (beta ** 2) * zetaA * zetaB) -((mA ** 4) * mB * beta * (1 + (4 -4 * 
        beta + 3 * (beta ** 2) * zetaA) * zetaB -4 * (-1 + beta) * (beta ** 
        2) * zetaA * (zetaB ** 2) + (mB ** 2) * (-1 + 4 * beta * zetaB -3 * 
        (beta ** 2) * zetaA * zetaB + 4 * (beta ** 3) * zetaA * (zetaB ** 
        2)))) + (mA ** 2) * mB * (-1 -((beta ** 2) * (4 + 5 * zetaA) * zetaB) 
        + 2 * (beta ** 4) * (-2 + zetaA) * zetaA * (zetaB ** 2) -((beta ** 5) 
        * (zetaA ** 3) * (zetaB ** 2)) + (beta ** 3) * zetaA * zetaB * (3 + 4 
        * zetaB -2 * zetaA * zetaB) + beta * (1 + zetaA + 4 * zetaB + 2 * 
        zetaA * zetaB) + (mB ** 4) * (beta ** 3) * (zetaA ** 2) * (3 + (beta ** 
        2) * zetaA * zetaB) + (mB ** 2) * (1 -(beta * (1 + zetaA)) + (beta ** 
        2) * (4 + 5 * zetaA) * zetaB + (beta ** 5) * (zetaA ** 3) * (-1 + 
        zetaB) * zetaB -2 * (beta ** 4) * (-2 + zetaA) * zetaA * (zetaB ** 2) 
        -3 * (beta ** 3) * zetaA * (zetaA + zetaB))) -(mB * zetaA * ((mB ** 
        4) * (-1 + beta) * (beta ** 2) * zetaA * (3 + (beta ** 2) * zetaA * 
        zetaB) -((-1 + beta) * (-1 + 2 * beta * zetaB -2 * (beta ** 3) * 
        zetaA * (zetaB ** 2) + (beta ** 4) * (zetaA ** 2) * (zetaB ** 2))) + 
        (mB ** 2) * (1 -beta -3 * (beta ** 3) * zetaA + (beta ** 5) * (zetaA ** 
        2) * (-1 + zetaB) * zetaB + (beta ** 2) * (3 * zetaA + 2 * zetaB) + 
        (beta ** 4) * zetaA * zetaB * (zetaA -2 * zetaB -(zetaA * zetaB))))) 
        + (mA ** 3) * ((mB ** 4) * (beta -4 * (beta ** 2) * zetaA + 3 * (beta 
        ** 3) * zetaA * zetaB -4 * (beta ** 4) * (zetaA ** 2) * zetaB) + (mB ** 
        2) * (1 -2 * (beta ** 4) * (zetaA ** 2) * (-2 + zetaB) * zetaB + 
        (beta ** 5) * (-1 + zetaA) * zetaA * (zetaB ** 3) -(beta * (1 + 
        zetaB)) -3 * (beta ** 3) * zetaB * (zetaA + zetaB) + (beta ** 2) * 
        zetaA * (4 + 5 * zetaB)) + zetaB * (-1 + beta + 3 * (beta ** 3) * 
        zetaB -((beta ** 5) * (-1 + zetaA) * zetaA * (zetaB ** 2)) -((beta ** 
        2) * (2 * zetaA + 3 * zetaB)) + (beta ** 4) * zetaA * zetaB * (-zetaB 
        + zetaA * (2 + zetaB)))) + mA * ((-1 + beta) * zetaB * (-1 + 2 * beta 
        * zetaA -2 * (beta ** 3) * (zetaA ** 2) * zetaB + (beta ** 4) * 
        (zetaA ** 2) * (zetaB ** 2)) + (mB ** 4) * beta * (-1 + 4 * (-1 + 
        beta) * (beta ** 2) * (zetaA ** 2) * zetaB + zetaA * (-4 + 4 * beta 
        -3 * (beta ** 2) * zetaB)) + (mB ** 2) * (-1 + (beta ** 3) * zetaA * 
        (3 -2 * zetaA * (-2 + zetaB)) * zetaB + 2 * (beta ** 4) * (zetaA ** 
        2) * (-2 + zetaB) * zetaB -((beta ** 5) * (zetaA ** 2) * (zetaB ** 
        3)) -((beta ** 2) * zetaA * (4 + 5 * zetaB)) + beta * (1 + zetaB + 2 
        * zetaA * (2 + zetaB))))))) / zetaB) / zetaA) / (-1 + (-1 + mA ** 2) 
        * (-1 + mB ** 2) * (beta ** 2))
        Γ1_22 = ((((0.125 * ((mA -mB) ** -2) * ((-1 + 4 * (beta ** 2) * zetaA * 
        zetaB) ** -3) * (-4 * (mB ** 4) * beta * (2 * beta * zetaB + 64 * 
        (beta ** 6) * (-1 + 4 * beta) * (zetaA ** 5) * (zetaB ** 2) + zetaA * 
        (1 -4 * beta * zetaB -24 * (beta ** 2) * zetaB + 32 * (beta ** 3) * 
        (zetaB ** 2)) + 8 * beta * (zetaA ** 2) * (-1 + 3 * beta * zetaB + 
        (beta ** 2) * (17 -6 * zetaB) * zetaB -28 * (beta ** 3) * (zetaB ** 
        2) + 12 * (beta ** 4) * (zetaB ** 3)) + 64 * (beta ** 4) * (zetaA ** 
        4) * zetaB * (-1 + 6 * (beta ** 3) * (zetaB ** 2) + beta * (4 + 
        zetaB) -2 * (beta ** 2) * zetaB * (5 + zetaB)) + 4 * (beta ** 2) * 
        (zetaA ** 3) * (-3 + 12 * (beta ** 2) * (-10 + zetaB) * zetaB + 160 * 
        (beta ** 3) * (zetaB ** 2) -64 * (beta ** 4) * (zetaB ** 3) + 4 * 
        beta * (3 + 5 * zetaB))) + (-1 + 2 * beta) * ((1 -2 * beta) * (zetaB ** 
        2) + 64 * (beta ** 6) * (1 + 6 * beta) * (zetaA ** 5) * (zetaB ** 3) 
        + 16 * (beta ** 4) * (zetaA ** 4) * (zetaB ** 2) * (7 + 8 * (beta ** 
        2) * (-14 + zetaB) * zetaB + 48 * (beta ** 3) * (zetaB ** 2) -6 * 
        beta * (-3 + 4 * zetaB)) + 2 * zetaA * zetaB * (1 + 20 * (beta ** 3) 
        * (zetaB ** 2) -2 * beta * (1 + 6 * zetaB) + 2 * (beta ** 2) * zetaB 
        * (4 + 7 * zetaB)) + 4 * (beta ** 2) * (zetaA ** 3) * zetaB * (7 + 
        beta * (10 -80 * zetaB) + 16 * (beta ** 3) * (41 -6 * zetaB) * (zetaB 
        ** 2) + 16 * (beta ** 4) * (-28 + zetaB) * (zetaB ** 3) + 96 * (beta ** 
        5) * (zetaB ** 4) + 8 * (beta ** 2) * zetaB * (-20 + 23 * zetaB)) + 
        (zetaA ** 2) * (1 + 288 * (beta ** 5) * (zetaB ** 4) -80 * (beta ** 
        3) * (zetaB ** 2) * (-1 + 4 * zetaB) + 16 * (beta ** 4) * (zetaB ** 
        3) * (-40 + 7 * zetaB) -2 * beta * (1 + 12 * zetaB) + 8 * (beta ** 2) 
        * zetaB * (2 + 23 * zetaB))) + 4 * (mA ** 4) * beta * (-zetaB + 128 * 
        (-1 + mB ** 2) * (beta ** 7) * (zetaA ** 2) * (zetaB ** 4) * (3 * 
        zetaA + 2 * zetaB) + 2 * beta * (zetaA * (-1 + mB ** 2 -6 * zetaB) -4 
        * (zetaB ** 2)) + 64 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 3) * 
        (zetaB * (18 -18 * (mB ** 2) + zetaB) + 2 * zetaA * (6 -6 * (mB ** 2) 
        + zetaB)) -32 * (beta ** 5) * zetaA * (zetaB ** 2) * (-8 * (-1 + mB ** 
        2) * (zetaB ** 2) + 2 * zetaA * zetaB * (26 -26 * (mB ** 2) + 5 * 
        zetaB) + (zetaA ** 2) * (3 -3 * (mB ** 2) + 8 * zetaB)) -8 * (beta ** 
        3) * zetaB * (-6 * (-1 + mB ** 2) * (zetaB ** 2) + (zetaA ** 2) * (4 
        -4 * (mB ** 2) + 10 * zetaB) + zetaA * zetaB * (49 -49 * (mB ** 2) + 
        26 * zetaB)) + 4 * (beta ** 2) * zetaB * (zetaB * (8 -8 * (mB ** 2) + 
        3 * zetaB) + zetaA * (14 -14 * (mB ** 2) + 26 * zetaB)) + 16 * (beta ** 
        4) * zetaA * (zetaB ** 2) * (2 * zetaB * (23 -23 * (mB ** 2) + 2 * 
        zetaB) + zetaA * (30 -30 * (mB ** 2) + 29 * zetaB))) -4 * (mB ** 2) * 
        beta * (-(beta * zetaB * (2 + zetaB)) + zetaA * (-1 + (1 + 2 * beta + 
        24 * (beta ** 2)) * zetaB + 8 * (1 -4 * beta) * (beta ** 2) * (zetaB ** 
        2) + 20 * (beta ** 3) * (zetaB ** 3)) + 64 * (beta ** 6) * (zetaA ** 
        5) * (zetaB ** 2) * (1 -zetaB + beta * (-4 + 3 * zetaB)) + (zetaA ** 
        2) * (1 + beta * (7 -8 * zetaB) + 4 * (beta ** 2) * (-4 + zetaB) * 
        zetaB + 48 * (beta ** 5) * (zetaB ** 3) * (-2 + 3 * zetaB) -32 * 
        (beta ** 4) * (zetaB ** 2) * (-7 + 10 * zetaB) + 8 * (beta ** 3) * 
        zetaB * (-17 + 11 * zetaB)) + 4 * (beta ** 2) * (zetaA ** 3) * (3 + 
        zetaB + 48 * (beta ** 5) * (zetaB ** 5) -3 * beta * (4 + 5 * zetaB) 
        -32 * (beta ** 4) * (zetaB ** 3) * (-2 + 7 * zetaB) + 8 * (beta ** 3) 
        * (zetaB ** 2) * (-20 + 41 * zetaB) -4 * (beta ** 2) * zetaB * (-30 + 
        23 * zetaB + zetaB ** 2)) + 16 * (beta ** 4) * (zetaA ** 4) * zetaB * 
        (4 -zetaB + 24 * (beta ** 3) * (-1 + zetaB) * (zetaB ** 2) -4 * (beta 
        ** 2) * zetaB * (-10 + 12 * zetaB + zetaB ** 2) + beta * (-16 + 5 * 
        zetaB + 8 * (zetaB ** 2)))) -4 * (mA ** 3) * mB * beta * (128 * (-1 + 
        mB ** 2) * (beta ** 6) * (zetaA ** 4) * (zetaB ** 2) * (3 + beta * 
        zetaB) + zetaB * (-3 -12 * (beta ** 2) * (zetaB ** 2) + beta * (6 -6 
        * (mB ** 2) + 8 * zetaB)) -2 * zetaA * (1 + beta * (-3 + 3 * (mB ** 
        2) -10 * zetaB) + 16 * (beta ** 4) * (zetaB ** 3) * (-3 + 3 * (mB ** 
        2) + 2 * zetaB) + 8 * (beta ** 2) * zetaB * (3 -3 * (mB ** 2) + 5 * 
        zetaB) -4 * (beta ** 3) * (zetaB ** 2) * (1 -(mB ** 2) + 14 * zetaB)) 
        + 32 * (beta ** 4) * (zetaA ** 3) * zetaB * (-5 + (-3 + 29 * beta) * 
        zetaB -4 * (beta ** 3) * (zetaB ** 3) + (mB ** 2) * (5 -29 * beta * 
        zetaB + 4 * (beta ** 3) * (zetaB ** 3))) -8 * (beta ** 2) * (zetaA ** 
        2) * (4 + (4 -33 * beta) * zetaB + 6 * beta * (-5 + 12 * beta) * 
        (zetaB ** 2) + 2 * (beta ** 2) * (17 + 6 * beta) * (zetaB ** 3) -8 * 
        (beta ** 3) * (3 + 2 * beta) * (zetaB ** 4) + 8 * (beta ** 4) * 
        (zetaB ** 5) + (mB ** 2) * (-4 + 33 * beta * zetaB -72 * (beta ** 2) 
        * (zetaB ** 2) -12 * (beta ** 3) * (zetaB ** 3) + 16 * (beta ** 4) * 
        (zetaB ** 4)))) + 4 * mA * mB * beta * (64 * (-1 + mB ** 2) * (beta ** 
        6) * (zetaA ** 5) * (zetaB ** 2) + 64 * (-1 + mB ** 2) * (beta ** 4) 
        * (zetaA ** 4) * zetaB * (1 -7 * beta * zetaB + 6 * (beta ** 2) * 
        zetaB + 2 * (beta ** 3) * (zetaB ** 2)) + zetaB * (-5 + (mB ** 2) * 
        (2 -6 * beta) -12 * (beta ** 2) * (zetaB ** 2) + beta * (6 + 8 * 
        zetaB)) + zetaA * (-5 + 32 * (beta ** 4) * (3 -2 * zetaB) * (zetaB ** 
        3) -16 * (beta ** 2) * zetaB * (3 + 7 * zetaB) + 8 * (beta ** 3) * 
        (zetaB ** 2) * (1 + 14 * zetaB) + beta * (6 + 56 * zetaB) + (mB ** 2) 
        * (3 -8 * (beta ** 3) * (zetaB ** 2) -96 * (beta ** 4) * (zetaB ** 3) 
        + 16 * (beta ** 2) * zetaB * (3 + 2 * zetaB) -6 * beta * (1 + 6 * 
        zetaB))) -8 * beta * (zetaA ** 2) * (-3 + 12 * (beta ** 4) * (1 -2 * 
        zetaB) * (zetaB ** 3) + 8 * (beta ** 5) * (-2 + zetaB) * (zetaB ** 4) 
        + 2 * (beta ** 3) * (zetaB ** 2) * (36 + 23 * zetaB) + beta * (4 + 30 
        * zetaB) -((beta ** 2) * zetaB * (33 + 76 * zetaB)) + (mB ** 2) * (3 
        -12 * (beta ** 4) * (zetaB ** 3) + 16 * (beta ** 5) * (zetaB ** 4) 
        -12 * (beta ** 3) * (zetaB ** 2) * (6 + zetaB) -2 * beta * (2 + 13 * 
        zetaB) + (beta ** 2) * zetaB * (33 + 46 * zetaB))) + 4 * (beta ** 2) 
        * (zetaA ** 3) * (-3 + 60 * beta * zetaB -32 * (beta ** 5) * (zetaB ** 
        4) + 8 * (beta ** 3) * (zetaB ** 2) * (29 + 8 * zetaB) -20 * (beta ** 
        2) * zetaB * (2 + 11 * zetaB) + (mB ** 2) * (3 -60 * beta * zetaB + 
        32 * (beta ** 5) * (zetaB ** 4) -8 * (beta ** 3) * (zetaB ** 2) * (29 
        + 8 * zetaB) + 4 * (beta ** 2) * zetaB * (10 + 49 * zetaB)))) + 4 * 
        (mA ** 2) * beta * (-(zetaB * (-1 + zetaA + zetaB)) + 64 * (-1 + mB ** 
        2) * (beta ** 7) * (zetaA ** 2) * (zetaB ** 2) * (2 * (mB ** 2) * 
        (zetaA ** 2) * (2 * zetaA + 3 * zetaB) + zetaB * (3 * (zetaA ** 3) + 
        6 * (zetaA ** 2) * zetaB + 3 * zetaA * (-2 + zetaB) * zetaB -4 * 
        (zetaB ** 2))) + beta * ((zetaA ** 2) * (1 -(mB ** 2) + 8 * zetaB) -2 
        * zetaA * (-1 -7 * zetaB -8 * (zetaB ** 2) + (mB ** 2) * (1 + zetaB)) 
        + zetaB * (2 * (mB ** 4) + 9 * zetaB -((mB ** 2) * (2 + zetaB)))) -4 
        * (beta ** 2) * zetaB * (zetaB * (8 -8 * (mB ** 2) + 3 * zetaB) + 
        (zetaA ** 2) * (2 -2 * (mB ** 2) + 17 * zetaB) + zetaA * (14 + 6 * 
        (mB ** 4) + 28 * zetaB + zetaB ** 2 -2 * (mB ** 2) * (10 + zetaB))) + 
        4 * (beta ** 3) * (2 * (mB ** 4) * zetaA * (6 * (zetaA ** 2) + 17 * 
        zetaA * zetaB + 4 * (zetaB ** 2)) + (mB ** 2) * (-12 * (zetaB ** 3) + 
        zetaA * (zetaB ** 2) * (-106 + 5 * zetaB) + 2 * (zetaA ** 2) * zetaB 
        * (-21 + 5 * zetaB) + (zetaA ** 3) * (-12 + 5 * zetaB)) + zetaB * (-5 
        * (zetaA ** 3) + 12 * (zetaB ** 2) + 2 * (zetaA ** 2) * (4 + 5 * 
        zetaB) + zetaA * zetaB * (98 + 47 * zetaB))) + 16 * (beta ** 4) * 
        zetaA * zetaB * (-2 * (mB ** 4) * zetaA * (15 * zetaA + 7 * zetaB) + 
        (mB ** 2) * ((zetaA ** 2) * (30 -20 * zetaB) + 4 * zetaA * (11 -5 * 
        zetaB) * zetaB + 46 * (zetaB ** 2)) + zetaB * (-2 * zetaB * (23 + 2 * 
        zetaB) + (zetaA ** 2) * (20 + 17 * zetaB) + zetaA * (-30 -9 * zetaB + 
        zetaB ** 2))) + 64 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 2) * (-2 
        * (mB ** 4) * zetaA * (5 * zetaA + 2 * zetaB) -2 * (mB ** 2) * (-9 * 
        (zetaB ** 2) + zetaA * zetaB * (-8 + 7 * zetaB) + (zetaA ** 2) * (-5 
        + 7 * zetaB)) + zetaB * ((zetaA ** 2) * (14 + zetaB) -(zetaB * (18 + 
        zetaB)) + zetaA * (-12 + 12 * zetaB + zetaB ** 2))) + 16 * (beta ** 
        5) * zetaA * zetaB * (2 * (mB ** 4) * zetaA * (8 * (zetaA ** 2) + 20 
        * zetaA * zetaB + 3 * (zetaB ** 2)) + (mB ** 2) * (-16 * (zetaB ** 3) 
        + zetaA * (zetaB ** 2) * (-110 + 9 * zetaB) + (zetaA ** 3) * (-16 + 9 
        * zetaB) + 2 * (zetaA ** 2) * zetaB * (-23 + 41 * zetaB)) + zetaB * 
        (16 * (zetaB ** 2) -((zetaA ** 3) * (9 + 8 * zetaB)) + zetaA * zetaB 
        * (104 + 11 * zetaB) -2 * (zetaA ** 2) * (-3 + 33 * zetaB + 8 * 
        (zetaB ** 2))))))) / zetaB) / zetaA) / (-1 + 4 * (-1 + mA ** 2) * (-1 
        + mB ** 2) * (beta ** 2))) / beta
        Γ2_11 = ((((0.5 * ((mA -mB) ** -2) * ((-1 + (beta ** 2) * zetaA * zetaB) ** 
        -3) * (-2 * (mA ** 7) * (beta ** 3) * (zetaB ** 2) * (8 + (3 -6 * 
        beta + 10 * (beta ** 2) * zetaA) * zetaB + 2 * (beta ** 2) * zetaA * 
        (2 -4 * beta + 3 * (beta ** 2) * zetaA) * (zetaB ** 2) + (1 -2 * 
        beta) * (beta ** 4) * (zetaA ** 2) * (zetaB ** 3) + 2 * (mB ** 2) * 
        (-4 + 3 * beta * zetaB -5 * (beta ** 2) * zetaA * zetaB + 4 * (beta ** 
        3) * zetaA * (zetaB ** 2) -3 * (beta ** 4) * (zetaA ** 2) * (zetaB ** 
        2) + (beta ** 5) * (zetaA ** 2) * (zetaB ** 3))) + 2 * (mA ** 6) * mB 
        * (beta ** 2) * zetaB * (3 + 3 * (4 + 11 * (beta ** 2) * zetaA) * 
        zetaB + beta * (-3 + 30 * beta * zetaA -30 * (beta ** 2) * zetaA + 29 
        * (beta ** 3) * (zetaA ** 2)) * (zetaB ** 2) -((beta ** 3) * zetaA * 
        (4 -14 * beta * zetaA + 10 * (beta ** 2) * zetaA + (beta ** 3) * 
        (zetaA ** 2)) * (zetaB ** 3)) -((beta ** 5) * (zetaA ** 2) * (zetaB ** 
        4)) + (mB ** 2) * (-3 -33 * (beta ** 2) * zetaA * zetaB + 30 * (beta ** 
        3) * zetaA * (zetaB ** 2) -29 * (beta ** 4) * (zetaA ** 2) * (zetaB ** 
        2) + 10 * (beta ** 5) * (zetaA ** 2) * (zetaB ** 3) + (beta ** 6) * 
        (zetaA ** 3) * (zetaB ** 3))) + (mA ** 4) * mB * beta * (-2 * (mB ** 
        4) * beta * zetaA * (1 -8 * beta * zetaB + 53 * (beta ** 2) * zetaA * 
        zetaB -24 * (beta ** 3) * zetaA * (zetaB ** 2) + 55 * (beta ** 4) * 
        (zetaA ** 2) * (zetaB ** 2) + 3 * (beta ** 6) * (zetaA ** 3) * (zetaB 
        ** 3)) + zetaB * (4 + (beta ** 3) * zetaA * (-66 + zetaA -60 * zetaB) 
        * zetaB + (beta ** 7) * (zetaA ** 3) * (2 + 3 * zetaA) * (zetaB ** 3) 
        + 2 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 3) * (10 -3 * zetaA + 
        zetaB) + 4 * (beta ** 4) * zetaA * (zetaB ** 2) * (15 -2 * zetaA + 2 
        * zetaB) + 2 * (beta ** 2) * zetaB * (13 * zetaA + 3 * zetaB) + (beta 
        ** 5) * (zetaA ** 2) * (zetaB ** 2) * (-58 + 73 * zetaA -28 * zetaB + 
        16 * zetaA * zetaB) -(beta * (6 + 13 * zetaA + 24 * zetaB + 16 * 
        zetaA * zetaB))) + (mB ** 2) * (-2 * zetaB + 2 * (beta ** 6) * (-10 + 
        zetaA) * (zetaA ** 2) * (zetaB ** 4) -2 * (beta ** 2) * zetaA * zetaB 
        * (8 + 7 * zetaB) + (beta ** 5) * (zetaA ** 2) * (zetaB ** 2) * (110 
        * zetaA + 58 * zetaB -57 * zetaA * zetaB) + (beta ** 7) * (zetaA ** 
        3) * (zetaB ** 3) * (6 * zetaA -2 * zetaB -3 * zetaA * zetaB) + 2 * 
        (beta ** 4) * zetaA * (zetaB ** 2) * (-24 * zetaA -30 * zetaB + 7 * 
        zetaA * zetaB) + beta * (2 * zetaA + 6 * zetaB + 49 * zetaA * zetaB) 
        + (beta ** 3) * zetaA * zetaB * (106 * zetaA + 66 * zetaB + 91 * 
        zetaA * zetaB))) + mA * (mB ** 2) * zetaA * (zetaB * (1 -14 * beta * 
        zetaA + (beta ** 2) * (1 + zetaA * (8 -9 * zetaB)) + 30 * (beta ** 7) 
        * (zetaA ** 3) * (zetaB ** 2) + (beta ** 6) * (zetaA ** 2) * (7 + 
        zetaA * (-8 + zetaB)) * (zetaB ** 2) -3 * (beta ** 8) * (zetaA ** 3) 
        * (zetaB ** 3) + 2 * (beta ** 5) * (zetaA ** 2) * zetaB * (26 + (20 
        -7 * zetaA) * zetaB) + (beta ** 4) * zetaA * zetaB * (11 -9 * zetaA * 
        zetaB) -2 * (beta ** 3) * zetaA * (1 + (20 + 26 * zetaA) * zetaB)) -2 
        * (mB ** 4) * (beta ** 2) * (-3 + (beta ** 5) * (zetaA ** 4) * (zetaB 
        ** 2) -(zetaA * (4 -8 * beta + (beta ** 2) * zetaB)) + (beta ** 3) * 
        (zetaA ** 3) * zetaB * (4 -6 * beta * zetaB + 18 * (beta ** 2) * 
        zetaB + (beta ** 3) * (zetaB ** 2)) + beta * (zetaA ** 2) * (3 -14 * 
        beta * zetaB + 46 * (beta ** 2) * zetaB + 3 * (beta ** 3) * (zetaB ** 
        2))) + (mB ** 2) * beta * (4 + 4 * (beta ** 4) * (zetaA ** 2) * (23 + 
        2 * zetaA -10 * zetaB) * zetaB + 2 * (beta ** 6) * (zetaA ** 3) * (18 
        + zetaA -15 * zetaB) * (zetaB ** 2) + (beta ** 3) * zetaA * zetaB * 
        (-2 -28 * zetaA + zetaB) + (beta ** 7) * (zetaA ** 3) * (zetaB ** 3) 
        * (2 + 3 * zetaB) + 2 * (beta ** 2) * zetaA * (8 + 3 * zetaA + 9 * 
        zetaB) -(beta * (6 -3 * zetaB + 8 * zetaA * (1 + zetaB))) + (beta ** 
        5) * (zetaA ** 2) * (zetaB ** 2) * (6 -7 * zetaB + 4 * zetaA * (-3 + 
        2 * zetaB)))) + (mB ** 3) * zetaA * (-2 * beta * zetaB + zetaA * (1 + 
        2 * beta * (-1 + mB ** 2 + zetaB) + 2 * (beta ** 3) * zetaB * (-1 + 
        mB ** 2 + 3 * zetaB) + (beta ** 2) * (1 -(mB ** 2) + 8 * zetaB)) + 
        (beta ** 6) * (zetaA ** 4) * (zetaB ** 2) * (2 * (mB ** 4) * beta * 
        (-1 + 2 * beta) + zetaB + 2 * beta * zetaB -3 * (beta ** 2) * zetaB + 
        (mB ** 2) * beta * (2 -2 * zetaB + beta * (-4 + 3 * zetaB))) + (beta ** 
        2) * (zetaA ** 2) * (6 * (mB ** 4) * beta * (-1 + 2 * beta + 2 * 
        (beta ** 2) * zetaB) -(zetaB * (-7 + 5 * (beta ** 2) + beta * (2 -12 
        * zetaB) + 6 * (beta ** 3) * zetaB * (2 + zetaB))) + (mB ** 2) * beta 
        * (12 * (beta ** 2) * (-1 + zetaB) * zetaB + 2 * (3 + zetaB) + beta * 
        (-12 + 5 * zetaB))) + (beta ** 4) * (zetaA ** 3) * zetaB * (4 * (mB ** 
        4) * beta * (-2 + 4 * beta + (beta ** 2) * zetaB) + zetaB * (7 + 2 * 
        (beta ** 3) * (-1 + zetaB) * zetaB + 2 * beta * (1 + zetaB) -((beta ** 
        2) * (9 + 8 * zetaB))) + (mB ** 2) * beta * (8 -2 * zetaB + 2 * (beta 
        ** 2) * (-2 + zetaB) * zetaB + beta * (-16 + 9 * zetaB)))) + (mA ** 
        3) * (2 * (mB ** 6) * (beta ** 2) * zetaA * (-3 + 8 * beta * zetaA 
        -((beta ** 2) * zetaA * zetaB) + 46 * (beta ** 3) * (zetaA ** 2) * 
        zetaB + 3 * (beta ** 4) * (zetaA ** 2) * (zetaB ** 2) + 18 * (beta ** 
        5) * (zetaA ** 3) * (zetaB ** 2) + (beta ** 6) * (zetaA ** 3) * 
        (zetaB ** 3)) + zetaB * (zetaB -3 * (beta ** 8) * (zetaA ** 3) * 
        (zetaB ** 4) + 2 * (beta ** 7) * (zetaA ** 3) * (zetaB ** 3) * (11 + 
        zetaA + zetaB) + 2 * (beta ** 5) * (zetaA ** 2) * (zetaB ** 2) * (10 
        -3 * zetaA + zetaB -7 * zetaA * zetaB) + (beta ** 6) * (zetaA ** 2) * 
        (zetaB ** 3) * (-9 + zetaA * zetaB) + (beta ** 4) * zetaA * (zetaB ** 
        2) * (-5 + 7 * zetaA * zetaB) + (beta ** 2) * zetaB * (1 + 7 * zetaA 
        * zetaB) -2 * beta * (zetaA + zetaB + 7 * zetaA * zetaB) -2 * (beta ** 
        3) * zetaA * zetaB * (-3 -3 * zetaA + zetaB + 10 * zetaA * zetaB)) + 
        (mB ** 2) * beta * ((beta ** 6) * (zetaA ** 4) * (zetaB ** 3) * (-26 
        + (-2 + 3 * beta) * zetaB) + zetaA * (-2 + (2 -21 * beta + 48 * (beta 
        ** 2)) * zetaB + 14 * (7 -3 * beta) * (beta ** 2) * (zetaB ** 2) -15 
        * (beta ** 3) * (zetaB ** 3)) + (beta ** 4) * (zetaA ** 3) * (zetaB ** 
        2) * (6 + (-34 -7 * beta + 32 * (beta ** 2)) * zetaB -2 * (beta ** 2) 
        * (11 + 3 * beta) * (zetaB ** 2) + 3 * (beta ** 3) * (zetaB ** 3)) + 
        (beta ** 2) * (zetaA ** 2) * zetaB * (54 + (34 -71 * beta + 112 * 
        (beta ** 2)) * zetaB + 2 * (39 -23 * beta) * (beta ** 2) * (zetaB ** 
        2) + 5 * (beta ** 3) * (zetaB ** 3)) + zetaB * (6 + beta * (-2 + 7 * 
        zetaB))) -((mB ** 4) * beta * (-2 * beta * zetaB + (beta ** 6) * 
        (zetaA ** 4) * (zetaB ** 2) * (36 + 2 * (-13 + beta) * zetaB + 3 * 
        beta * (zetaB ** 2)) + (beta ** 2) * (zetaA ** 2) * (16 -2 * (-35 + 
        beta) * zetaB + beta * (-59 + 112 * beta) * (zetaB ** 2) -46 * (beta ** 
        3) * (zetaB ** 3)) + (beta ** 4) * (zetaA ** 3) * zetaB * (92 + 6 * 
        (3 + beta) * zetaB + beta * (-7 + 32 * beta) * (zetaB ** 2) -6 * 
        (beta ** 3) * (zetaB ** 3)) + zetaA * (2 + 48 * (beta ** 2) * zetaB 
        -42 * (beta ** 3) * (zetaB ** 2) -(beta * (6 + 17 * zetaB)))))) + (mA 
        ** 2) * (-4 * (mB ** 7) * (beta ** 4) * (zetaA ** 3) * (3 + 3 * beta 
        * zetaB + 4 * (beta ** 2) * zetaA * zetaB + (beta ** 3) * zetaA * 
        (zetaB ** 2) + (beta ** 4) * (zetaA ** 2) * (zetaB ** 2)) + (mB ** 5) 
        * beta * zetaA * (-6 + beta * (2 + 9 * zetaA) -2 * (beta ** 2) * (8 + 
        21 * zetaA) * zetaB -2 * (beta ** 6) * (zetaA ** 3) * (-2 + zetaB) * 
        (zetaB ** 2) + (beta ** 7) * (zetaA ** 3) * (zetaB ** 2) * (4 * zetaA 
        + 6 * zetaB -3 * zetaA * zetaB) + (beta ** 5) * (zetaA ** 2) * zetaB 
        * (16 * zetaA + 110 * zetaB + 11 * zetaA * zetaB) -2 * (beta ** 4) * 
        zetaA * zetaB * (-6 * zetaA + 24 * zetaB + 23 * zetaA * zetaB) + 
        (beta ** 3) * zetaA * (12 * zetaA + 106 * zetaB + 47 * zetaA * 
        zetaB)) + mB * zetaA * zetaB * (1 + 2 * beta * zetaB + 6 * (beta ** 
        7) * (zetaA ** 2) * (zetaB ** 3) -3 * (beta ** 8) * (zetaA ** 3) * 
        (zetaB ** 3) + (beta ** 6) * (zetaA ** 2) * (zetaB ** 2) * (-89 + 
        (-16 + zetaA) * zetaB) + (beta ** 4) * zetaA * zetaB * (-21 + 55 * 
        zetaA * zetaB) + (beta ** 2) * (1 + (16 + 55 * zetaA) * zetaB) + 2 * 
        (beta ** 5) * zetaA * (zetaB ** 2) * (10 + zetaA * (4 + zetaB)) -2 * 
        (beta ** 3) * zetaB * (5 + 2 * zetaA * (2 + 5 * zetaB))) + (mB ** 3) 
        * beta * (-2 * zetaB + 3 * (beta ** 7) * (zetaA ** 5) * (zetaB ** 3) 
        + (beta ** 5) * (zetaA ** 4) * (zetaB ** 2) * (-11 + (8 + 2 * beta -6 
        * (beta ** 2)) * zetaB + beta * (-2 + 3 * beta) * (zetaB ** 2)) 
        -((beta ** 3) * (zetaA ** 3) * zetaB * (47 + (beta ** 2) * (110 -73 * 
        zetaB) * zetaB + 2 * beta * (-23 + zetaB) * zetaB + 2 * (beta ** 3) * 
        (zetaB ** 3))) -(beta * (zetaA ** 2) * (9 + 2 * (4 -21 * beta + 53 * 
        (beta ** 2)) * zetaB + beta * (-2 + 71 * beta -48 * (beta ** 2)) * 
        (zetaB ** 2) + 26 * (beta ** 3) * (zetaB ** 3))) + zetaA * (-2 * 
        (beta ** 2) * (-8 + zetaB) * zetaB + 2 * (3 + zetaB) -(beta * (2 + 37 
        * zetaB))))) -((mA ** 5) * beta * zetaB * (2 * (mB ** 4) * beta * (1 
        -24 * beta * zetaA + 21 * (beta ** 2) * zetaA * zetaB -56 * (beta ** 
        3) * (zetaA ** 2) * zetaB + 23 * (beta ** 4) * (zetaA ** 2) * (zetaB ** 
        2) -16 * (beta ** 5) * (zetaA ** 3) * (zetaB ** 2) + 3 * (beta ** 6) 
        * (zetaA ** 3) * (zetaB ** 3)) + zetaB * (-2 + beta + (beta ** 3) * 
        (12 -5 * zetaA) * zetaB + (beta ** 5) * (16 -9 * zetaA) * zetaA * 
        (zetaB ** 2) + (beta ** 7) * (4 -3 * zetaA) * (zetaA ** 2) * (zetaB ** 
        3) -2 * (beta ** 2) * (8 + zetaA * (-3 + zetaB) + 3 * zetaB) + 2 * 
        (beta ** 6) * (zetaA ** 2) * (zetaB ** 2) * (-6 -zetaB + zetaA * (11 
        + zetaB)) + 2 * (beta ** 4) * zetaA * zetaB * (zetaA * (10 + zetaB) 
        -2 * (5 + 2 * zetaB))) + (mB ** 2) * (6 + (beta ** 7) * (zetaA ** 2) 
        * (3 * zetaA * (-2 + zetaB) -4 * zetaB) * (zetaB ** 3) + 2 * (beta ** 
        6) * (zetaA ** 2) * (zetaB ** 2) * (zetaA * (16 -11 * zetaB) + 6 * 
        zetaB) + beta * (-2 + 7 * zetaB) + (beta ** 5) * zetaA * (zetaB ** 2) 
        * (-16 * zetaB + zetaA * (-46 + 5 * zetaB)) -3 * (beta ** 3) * zetaB 
        * (4 * zetaB + zetaA * (14 + 5 * zetaB)) + 2 * (beta ** 4) * zetaA * 
        zetaB * (10 * zetaB + zetaA * (56 + 39 * zetaB)) + 2 * (beta ** 2) * 
        (8 * zetaB + zetaA * (24 + 49 * zetaB))))))) / zetaB) / zetaA) / (-1 
        + (-1 + mA ** 2) * (-1 + mB ** 2) * (beta ** 2))) / beta
        Γ2_12 = (((0.5 * ((mA -mB) ** -2) * ((-1 + (beta ** 2) * zetaA * zetaB) ** 
        -3) * (-(((-4 * mA * mB * beta * zetaA * zetaB + (mB ** 2) * zetaA * 
        (1 + (beta ** 2) * zetaA * zetaB) + (mA ** 2) * zetaB * (1 + (beta ** 
        2) * zetaA * zetaB)) * (-2 * (mB ** 4) * (beta ** 2) * (-1 + beta * 
        zetaA) * (-1 + 3 * beta * zetaA -3 * (beta ** 2) * zetaA * zetaB + 
        (beta ** 3) * (zetaA ** 2) * zetaB) + 2 * (mA ** 3) * mB * beta * (-1 
        + beta * zetaB) * (-1 + 3 * beta * zetaB -3 * (beta ** 2) * zetaA * 
        zetaB + (beta ** 3) * zetaA * (zetaB ** 2)) + 2 * (mA ** 4) * (-1 + 
        mB ** 2) * (beta ** 2) * (-1 + beta * zetaB) * (-1 + 3 * beta * zetaB 
        -3 * (beta ** 2) * zetaA * zetaB + (beta ** 3) * zetaA * (zetaB ** 
        2)) + 2 * mA * mB * beta * (-2 + 4 * beta * (zetaA + zetaB) + 4 * 
        (beta ** 3) * zetaA * zetaB * (zetaA + zetaB) -3 * (beta ** 2) * 
        ((zetaA + zetaB) ** 2) + (mB ** 2) * (-1 + beta * zetaA) * (-1 + 3 * 
        beta * zetaA -3 * (beta ** 2) * zetaA * zetaB + (beta ** 3) * (zetaA ** 
        2) * zetaB) -((beta ** 4) * zetaA * zetaB * (zetaA ** 2 + zetaB ** 
        2))) + (-1 + beta ** 2) * (zetaB + (beta ** 4) * (zetaA ** 3) * 
        (zetaB ** 2) + (beta ** 2) * (zetaA ** 2) * zetaB * (6 -8 * beta * 
        zetaB + (beta ** 2) * (zetaB ** 2)) + zetaA * (1 -8 * beta * zetaB + 
        6 * (beta ** 2) * (zetaB ** 2))) -((mB ** 2) * (beta ** 2) * (-2 + 
        zetaB + (beta ** 4) * (zetaA ** 3) * (-2 + zetaB) * zetaB + zetaA * 
        (1 -8 * beta * (-1 + zetaB) + 6 * (beta ** 2) * (-1 + zetaB) * zetaB) 
        + (beta ** 2) * (zetaA ** 2) * (-6 + (6 + 8 * beta) * zetaB -8 * beta 
        * (zetaB ** 2) + (beta ** 2) * (zetaB ** 3)))) + (mA ** 2) * (-1 + mB 
        ** 2) * (beta ** 2) * (-2 + zetaB + 8 * beta * zetaB -6 * (beta ** 2) 
        * (zetaB ** 2) + (beta ** 4) * (zetaA ** 3) * (zetaB ** 2) + 2 * (mB ** 
        2) * (-1 + beta * zetaA) * (-1 + 3 * beta * zetaA -3 * (beta ** 2) * 
        zetaA * zetaB + (beta ** 3) * (zetaA ** 2) * zetaB) + (beta ** 2) * 
        (zetaA ** 2) * zetaB * (6 -8 * beta * zetaB + (beta ** 2) * (zetaB ** 
        2)) + zetaA * (1 -8 * beta * zetaB + 6 * (beta ** 2) * (-1 + zetaB) * 
        zetaB + 8 * (beta ** 3) * (zetaB ** 2) -2 * (beta ** 4) * (zetaB ** 
        3))))) / beta) + 2 * (-(mA * zetaB * (1 -2 * beta * zetaA + (beta ** 
        2) * zetaA * zetaB)) -(mB * zetaA * (1 -2 * beta * zetaB + (beta ** 
        2) * zetaA * zetaB))) * ((mA ** 5) * (beta ** 2) * (1 + (-1 + mB ** 
        2) * beta) * (zetaB ** 2) * (3 + (beta ** 2) * zetaA * zetaB) -((mA ** 
        4) * mB * beta * (1 + (4 -4 * beta + 3 * (beta ** 2) * zetaA) * zetaB 
        -4 * (-1 + beta) * (beta ** 2) * zetaA * (zetaB ** 2) + (mB ** 2) * 
        (-1 + 4 * beta * zetaB -3 * (beta ** 2) * zetaA * zetaB + 4 * (beta ** 
        3) * zetaA * (zetaB ** 2)))) + (mA ** 2) * mB * (-1 -((beta ** 2) * 
        (4 + 5 * zetaA) * zetaB) + 2 * (beta ** 4) * (-2 + zetaA) * zetaA * 
        (zetaB ** 2) -((beta ** 5) * (zetaA ** 3) * (zetaB ** 2)) + (beta ** 
        3) * zetaA * zetaB * (3 + 4 * zetaB -2 * zetaA * zetaB) + beta * (1 + 
        zetaA + 4 * zetaB + 2 * zetaA * zetaB) + (mB ** 4) * (beta ** 3) * 
        (zetaA ** 2) * (3 + (beta ** 2) * zetaA * zetaB) + (mB ** 2) * (1 
        -(beta * (1 + zetaA)) + (beta ** 2) * (4 + 5 * zetaA) * zetaB + (beta 
        ** 5) * (zetaA ** 3) * (-1 + zetaB) * zetaB -2 * (beta ** 4) * (-2 + 
        zetaA) * zetaA * (zetaB ** 2) -3 * (beta ** 3) * zetaA * (zetaA + 
        zetaB))) -(mB * zetaA * ((mB ** 4) * (-1 + beta) * (beta ** 2) * 
        zetaA * (3 + (beta ** 2) * zetaA * zetaB) -((-1 + beta) * (-1 + 2 * 
        beta * zetaB -2 * (beta ** 3) * zetaA * (zetaB ** 2) + (beta ** 4) * 
        (zetaA ** 2) * (zetaB ** 2))) + (mB ** 2) * (1 -beta -3 * (beta ** 3) 
        * zetaA + (beta ** 5) * (zetaA ** 2) * (-1 + zetaB) * zetaB + (beta ** 
        2) * (3 * zetaA + 2 * zetaB) + (beta ** 4) * zetaA * zetaB * (zetaA 
        -2 * zetaB -(zetaA * zetaB))))) + (mA ** 3) * ((mB ** 4) * (beta -4 * 
        (beta ** 2) * zetaA + 3 * (beta ** 3) * zetaA * zetaB -4 * (beta ** 
        4) * (zetaA ** 2) * zetaB) + (mB ** 2) * (1 -2 * (beta ** 4) * (zetaA 
        ** 2) * (-2 + zetaB) * zetaB + (beta ** 5) * (-1 + zetaA) * zetaA * 
        (zetaB ** 3) -(beta * (1 + zetaB)) -3 * (beta ** 3) * zetaB * (zetaA 
        + zetaB) + (beta ** 2) * zetaA * (4 + 5 * zetaB)) + zetaB * (-1 + 
        beta + 3 * (beta ** 3) * zetaB -((beta ** 5) * (-1 + zetaA) * zetaA * 
        (zetaB ** 2)) -((beta ** 2) * (2 * zetaA + 3 * zetaB)) + (beta ** 4) 
        * zetaA * zetaB * (-zetaB + zetaA * (2 + zetaB)))) + mA * ((-1 + 
        beta) * zetaB * (-1 + 2 * beta * zetaA -2 * (beta ** 3) * (zetaA ** 
        2) * zetaB + (beta ** 4) * (zetaA ** 2) * (zetaB ** 2)) + (mB ** 4) * 
        beta * (-1 + 4 * (-1 + beta) * (beta ** 2) * (zetaA ** 2) * zetaB + 
        zetaA * (-4 + 4 * beta -3 * (beta ** 2) * zetaB)) + (mB ** 2) * (-1 + 
        (beta ** 3) * zetaA * (3 -2 * zetaA * (-2 + zetaB)) * zetaB + 2 * 
        (beta ** 4) * (zetaA ** 2) * (-2 + zetaB) * zetaB -((beta ** 5) * 
        (zetaA ** 2) * (zetaB ** 3)) -((beta ** 2) * zetaA * (4 + 5 * zetaB)) 
        + beta * (1 + zetaB + 2 * zetaA * (2 + zetaB))))))) / zetaB) / zetaA) /\
        (-1 + (-1 + mA ** 2) * (-1 + mB ** 2) * (beta ** 2))
        Γ2_22 = ((((0.25 * ((mA -mB) ** -2) * ((-1 + 4 * (beta ** 2) * zetaA * 
        zetaB) ** -3) * (-4 * (mA ** 5) * beta * zetaB * (-1 -8 * beta * 
        zetaB + 256 * (-1 + mB ** 2) * (beta ** 7) * (zetaA ** 2) * (zetaB ** 
        4) + 64 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 3) * (14 -14 * (mB ** 
        2) + zetaB) -16 * (beta ** 3) * zetaB * (-2 * zetaA * (-6 + 6 * (mB ** 
        2) -5 * zetaB) -3 * (-1 + mB ** 2) * zetaB) -256 * (beta ** 5) * 
        zetaA * (zetaB ** 2) * (zetaB -((mB ** 2) * zetaB) + zetaA * (3 -3 * 
        (mB ** 2) + zetaB)) + (beta ** 2) * (-8 * zetaA * (-1 + mB ** 2 -6 * 
        zetaB) + 4 * zetaB * (8 -8 * (mB ** 2) + 3 * zetaB)) + 16 * (beta ** 
        4) * zetaA * zetaB * (2 * zetaB * (17 -17 * (mB ** 2) + 2 * zetaB) + 
        zetaA * (6 -6 * (mB ** 2) + 13 * zetaB))) + 4 * (mA ** 4) * mB * beta 
        * (-3 * zetaB + 128 * (-1 + mB ** 2) * (beta ** 7) * (zetaA ** 3) * 
        (zetaA -3 * zetaB) * (zetaB ** 3) -64 * (beta ** 6) * (zetaA ** 2) * 
        (zetaB ** 3) * (2 * zetaA * (8 -8 * (mB ** 2) + zetaB) + zetaB * (2 
        -2 * (mB ** 2) + zetaB)) -16 * (beta ** 4) * zetaA * (zetaB ** 2) * 
        (25 * zetaA * (2 -2 * (mB ** 2) + zetaB) + 2 * zetaB * (3 -3 * (mB ** 
        2) + 2 * zetaB)) + 32 * (beta ** 5) * (zetaA ** 2) * (zetaB ** 2) * 
        (zetaB * (29 -29 * (mB ** 2) + 6 * zetaB) + zetaA * (15 -15 * (mB ** 
        2) + 8 * zetaB)) + beta * (2 * zetaB * (3 -3 * (mB ** 2) + 4 * zetaB) 
        + zetaA * (2 -2 * (mB ** 2) + 20 * zetaB)) + 8 * (beta ** 3) * zetaA 
        * zetaB * (zetaB * (29 -29 * (mB ** 2) + 14 * zetaB) + zetaA * (17 
        -17 * (mB ** 2) + 22 * zetaB)) -4 * (beta ** 2) * (3 * (zetaB ** 3) + 
        zetaA * zetaB * (22 -22 * (mB ** 2) + 26 * zetaB))) -4 * (mA ** 2) * 
        mB * beta * ((-5 + 2 * (mB ** 2) -zetaA) * zetaB + 64 * (beta ** 6) * 
        (zetaA ** 2) * (zetaB ** 2) * (-6 * (mB ** 4) * (zetaA ** 2) + (mB ** 
        2) * ((zetaA ** 2) * (6 -11 * zetaB) + zetaA * (16 -3 * zetaB) * 
        zetaB + 2 * (zetaB ** 2)) + zetaB * (zetaA * (-16 + zetaB) -(zetaB * 
        (2 + zetaB)) + (zetaA ** 2) * (11 + zetaB))) + 64 * (-1 + mB ** 2) * 
        (beta ** 7) * (zetaA ** 3) * (zetaB ** 2) * (4 * (mB ** 2) * (zetaA ** 
        2) + zetaB * (3 * (zetaA ** 2) -6 * zetaB + zetaA * (2 + 3 * zetaB))) 
        -4 * (beta ** 2) * zetaB * (3 * (zetaB ** 2) + (zetaA ** 2) * (25 -25 
        * (mB ** 2) + 9 * zetaB) + zetaA * (22 + 2 * (mB ** 4) + 35 * zetaB 
        -3 * (mB ** 2) * (8 + 3 * zetaB))) -16 * (beta ** 4) * zetaA * zetaB 
        * (6 * (mB ** 4) * zetaA * (3 * zetaA + zetaB) + zetaB * ((zetaA ** 
        2) * (10 -9 * zetaB) + 25 * zetaA * (2 + zetaB) + 2 * zetaB * (3 + 2 
        * zetaB)) -2 * (mB ** 2) * (28 * zetaA * zetaB + 3 * (zetaB ** 2) + 
        (zetaA ** 2) * (9 + 5 * zetaB))) + beta * (2 * zetaB * (3 -3 * (mB ** 
        2) + 4 * zetaB) + (zetaA ** 2) * (1 -(mB ** 2) + 8 * zetaB) + zetaA * 
        (2 + 49 * zetaB + 4 * (zetaB ** 2) -((mB ** 2) * (2 + 29 * zetaB)))) 
        + 4 * (beta ** 3) * zetaA * (4 * (mB ** 4) * zetaA * (3 * zetaA + 4 * 
        zetaB) -((mB ** 2) * (58 * (zetaB ** 2) + (zetaA ** 2) * (12 + 7 * 
        zetaB) + zetaA * zetaB * (50 + 63 * zetaB))) + zetaB * (7 * (zetaA ** 
        2) + 2 * zetaB * (29 + 14 * zetaB) + zetaA * (34 + 107 * zetaB))) + 
        16 * (beta ** 5) * (zetaA ** 2) * zetaB * (16 * (mB ** 4) * zetaA * 
        (zetaA + zetaB) + (mB ** 2) * (-58 * (zetaB ** 2) + (zetaA ** 2) * 
        (-16 + 5 * zetaB) + zetaA * zetaB * (-46 + 25 * zetaB)) + zetaB * (2 
        * zetaB * (29 + 6 * zetaB) -((zetaA ** 2) * (5 + 8 * zetaB)) + zetaA 
        * (30 -9 * zetaB -4 * (zetaB ** 2))))) -4 * (mA ** 3) * beta * (2 * 
        (mB ** 4) * beta * (zetaB + 64 * (beta ** 5) * (zetaA ** 4) * (zetaB ** 
        2) * (-5 + 3 * beta * zetaB) + zetaA * (3 -12 * beta * zetaB + 4 * 
        (beta ** 2) * (zetaB ** 2)) -16 * (beta ** 3) * (zetaA ** 3) * zetaB 
        * (11 -29 * beta * zetaB + 4 * (beta ** 3) * (zetaB ** 3)) -4 * beta 
        * (zetaA ** 2) * (4 -29 * beta * zetaB + 36 * (beta ** 2) * (zetaB ** 
        2) + 4 * (beta ** 3) * (zetaB ** 3))) + (mB ** 2) * (64 * (beta ** 6) 
        * (zetaA ** 4) * (zetaB ** 2) * (10 -3 * (1 + 2 * beta) * zetaB + 3 * 
        beta * (zetaB ** 2)) -(beta * zetaB * (2 + zetaB -32 * beta * zetaB + 
        48 * (beta ** 2) * (zetaB ** 2))) -4 * (beta ** 2) * (zetaA ** 2) * 
        (-8 + (-9 + 58 * beta) * zetaB + (31 -96 * beta) * beta * (zetaB ** 
        2) + 8 * (beta ** 2) * (3 + 23 * beta) * (zetaB ** 3) -4 * (beta ** 
        3) * (5 + 56 * beta) * (zetaB ** 4) + 64 * (beta ** 5) * (zetaB ** 
        5)) + zetaA * (2 + 544 * (beta ** 4) * (zetaB ** 3) -256 * (beta ** 
        5) * (zetaB ** 4) -4 * (beta ** 3) * (zetaB ** 2) * (50 + 7 * zetaB) 
        + 4 * (beta ** 2) * zetaB * (8 + 9 * zetaB) -(beta * (6 + 13 * 
        zetaB))) + 16 * (beta ** 4) * (zetaA ** 3) * zetaB * (22 -44 * (beta ** 
        2) * (zetaB ** 3) + 4 * (beta ** 3) * (zetaB ** 3) * (2 + 3 * zetaB) 
        + beta * zetaB * (-58 + 41 * zetaB))) + zetaB * (1 -zetaB -64 * (beta 
        ** 7) * (zetaA ** 2) * (zetaB ** 3) * (3 * (zetaA ** 2) -4 * zetaB + 
        3 * zetaA * zetaB) + beta * (zetaA + 9 * zetaB + 12 * zetaA * zetaB) 
        + 64 * (beta ** 6) * (zetaA ** 2) * (zetaB ** 2) * (3 * (zetaA ** 2) 
        + zetaA * zetaB * (11 + zetaB) -(zetaB * (14 + zetaB))) -16 * (beta ** 
        5) * zetaA * (zetaB ** 2) * (-16 * zetaB -(zetaA * (48 + 7 * zetaB)) 
        + (zetaA ** 2) * (41 + 12 * zetaB)) + 4 * (beta ** 3) * zetaB * (-5 * 
        (zetaA ** 2) + 12 * zetaB + zetaA * (48 + 35 * zetaB)) + 16 * (beta ** 
        4) * zetaA * zetaB * (-2 * zetaB * (17 + 2 * zetaB) + (zetaA ** 2) * 
        (6 + 8 * zetaB) + zetaA * (-6 + zetaB + zetaB ** 2)) -4 * (beta ** 2) 
        * (zetaB * (8 + 3 * zetaB) + (zetaA ** 2) * (1 + 8 * zetaB) + zetaA * 
        (2 + 13 * zetaB + zetaB ** 2)))) + mA * (-((-1 + 2 * beta) * zetaB * 
        (zetaB -2 * beta * zetaB + 64 * (beta ** 5) * (1 + 6 * beta) * (zetaA 
        ** 4) * (zetaB ** 2) * (-1 + beta * zetaB) + zetaA * (1 + 40 * (beta ** 
        3) * (zetaB ** 2) + 4 * (beta ** 2) * zetaB * (2 + 7 * zetaB) -2 * 
        beta * (1 + 10 * zetaB)) + 16 * (beta ** 3) * (zetaA ** 3) * zetaB * 
        (-6 + 2 * (beta ** 2) * (41 -10 * zetaB) * zetaB + 4 * (beta ** 3) * 
        (-22 + zetaB) * (zetaB ** 2) + 24 * (beta ** 4) * (zetaB ** 3) + beta 
        * (-12 + 23 * zetaB)) + 4 * beta * (zetaA ** 2) * (-1 + 2 * (beta ** 
        2) * (5 -28 * zetaB) * zetaB + 28 * (beta ** 3) * (-4 + zetaB) * 
        (zetaB ** 2) + 72 * (beta ** 4) * (zetaB ** 3) + beta * (2 + 23 * 
        zetaB)))) + 4 * (mB ** 4) * beta * (2 * beta * zetaB -64 * (beta ** 
        6) * (zetaA ** 5) * (zetaB ** 2) + 64 * (beta ** 4) * (zetaA ** 4) * 
        zetaB * (-1 + 7 * beta * zetaB + 6 * (beta ** 3) * (zetaB ** 2) -2 * 
        (beta ** 2) * zetaB * (5 + zetaB)) + zetaA * (-3 -24 * (beta ** 2) * 
        zetaB + 8 * (beta ** 3) * (zetaB ** 2) + beta * (6 + 4 * zetaB)) -8 * 
        beta * (zetaA ** 2) * (-3 + 36 * (beta ** 3) * (zetaB ** 2) + 4 * 
        (beta ** 4) * (zetaB ** 3) -((beta ** 2) * zetaB * (29 + 6 * zetaB)) 
        + beta * (4 + 13 * zetaB)) -4 * (beta ** 2) * (zetaA ** 3) * (3 -60 * 
        beta * zetaB -232 * (beta ** 3) * (zetaB ** 2) + 32 * (beta ** 5) * 
        (zetaB ** 4) + 4 * (beta ** 2) * zetaB * (22 + 25 * zetaB))) + 4 * 
        (mB ** 2) * beta * (64 * (beta ** 6) * (zetaA ** 5) * (zetaB ** 2) 
        -(beta * zetaB * (2 + zetaB)) + zetaA * (5 + zetaB + 12 * (beta ** 2) 
        * zetaB * (2 + 3 * zetaB) -4 * (beta ** 3) * (zetaB ** 2) * (2 + 7 * 
        zetaB) -(beta * (6 + 17 * zetaB))) + 4 * (beta ** 2) * (zetaA ** 3) * 
        (3 -60 * beta * zetaB -176 * (beta ** 4) * (zetaB ** 4) + 16 * (beta ** 
        5) * (zetaB ** 4) * (2 + 3 * zetaB) + 4 * (beta ** 3) * (zetaB ** 2) 
        * (-58 + 41 * zetaB) -4 * (beta ** 2) * zetaB * (-22 -25 * zetaB + 
        zetaB ** 2)) + 64 * (beta ** 4) * (zetaA ** 4) * zetaB * (1 + beta * 
        (-7 + zetaB) * zetaB + 3 * (beta ** 3) * (-2 + zetaB) * (zetaB ** 2) 
        -((beta ** 2) * zetaB * (-10 + zetaB + zetaB ** 2))) + 4 * beta * 
        (zetaA ** 2) * (-6 -zetaB -24 * (beta ** 3) * (-3 + zetaB) * (zetaB ** 
        2) + 4 * (beta ** 4) * (zetaB ** 3) * (2 + 5 * zetaB) -((beta ** 2) * 
        zetaB * (58 + 43 * zetaB)) + beta * (8 + 35 * zetaB + zetaB ** 2)))) 
        + mB * zetaA * (4 * (mB ** 4) * beta * (1 -8 * beta * zetaA + 256 * 
        (beta ** 7) * (zetaA ** 4) * (zetaB ** 2) -64 * (beta ** 6) * (zetaA ** 
        3) * (6 + zetaA) * (zetaB ** 2) + 256 * (beta ** 5) * (zetaA ** 2) * 
        zetaB * (zetaA + zetaB) -16 * (beta ** 4) * zetaA * zetaB * (4 * 
        (zetaA ** 2) -3 * zetaA * (-6 + zetaB) + 6 * zetaB) -4 * (beta ** 2) 
        * (3 * (zetaA ** 2) + 2 * zetaB -4 * zetaA * zetaB) + 16 * (beta ** 
        3) * zetaA * (4 * zetaB + zetaA * (3 + 2 * zetaB))) -((-1 + 2 * beta) 
        * (64 * (beta ** 6) * (1 + 6 * beta) * (zetaA ** 4) * (zetaB ** 3) + 
        (-1 + 2 * beta) * zetaB * (-1 + 4 * beta * zetaB) + 16 * (beta ** 4) 
        * (zetaA ** 3) * (zetaB ** 2) * (7 + beta * (18 -20 * zetaB) + 4 * 
        (beta ** 2) * (-22 + zetaB) * zetaB + 24 * (beta ** 3) * (zetaB ** 
        2)) + zetaA * (1 + 8 * (beta ** 3) * (5 -12 * zetaB) * (zetaB ** 2) 
        -192 * (beta ** 4) * (zetaB ** 3) -2 * beta * (1 + 10 * zetaB) + 4 * 
        (beta ** 2) * zetaB * (2 + 23 * zetaB)) -4 * (beta ** 2) * (zetaA ** 
        2) * zetaB * (-7 + 4 * (beta ** 2) * (28 -23 * zetaB) * zetaB + 96 * 
        (beta ** 4) * (zetaB ** 3) + 8 * (beta ** 3) * (zetaB ** 2) * (-41 + 
        2 * zetaB) + 2 * beta * (-5 + 28 * zetaB)))) + 4 * (mB ** 2) * beta * 
        (-1 -(beta * zetaB) + 4 * (beta ** 2) * zetaB * (2 + zetaB) + 64 * 
        (beta ** 6) * (zetaA ** 4) * (zetaB ** 2) * (1 -zetaB + beta * (-4 + 
        3 * zetaB)) + zetaA * (1 + beta * (7 -4 * zetaB) -12 * (beta ** 2) * 
        zetaB -96 * (beta ** 4) * (-1 + zetaB) * (zetaB ** 2) + 4 * (beta ** 
        3) * zetaB * (-16 + 5 * zetaB)) -4 * (beta ** 2) * (zetaA ** 2) * (-3 
        -zetaB + 4 * (beta ** 3) * (16 -41 * zetaB) * (zetaB ** 2) + 48 * 
        (beta ** 4) * (zetaB ** 4) + 3 * beta * (4 + zetaB) + 4 * (beta ** 2) 
        * zetaB * (-18 + 17 * zetaB)) + 16 * (beta ** 4) * (zetaA ** 3) * 
        zetaB * (4 -zetaB + 4 * (beta ** 2) * (6 -11 * zetaB) * zetaB + 12 * 
        (beta ** 3) * (zetaB ** 3) + beta * (-16 + 9 * zetaB + 4 * (zetaB ** 
        2))))))) / zetaB) / zetaA) / (-1 + 4 * (-1 + mA ** 2) * (-1 + mB ** 
        2) * (beta ** 2))) / beta
        return np.array([[[Γ1_11, Γ1_12],
                          [Γ1_12, Γ1_22]],
                         [[Γ2_11, Γ2_12],
                          [Γ2_12, Γ2_22]]])
    
    def phase_transition_line(self, β):
        return self._phase_transition_line(1/β) * β
    
    def is_ordered_phase(self, x):
        return super().is_ordered_phase((1/x[0], x[1]/x[0]))
    
if __name__ == "__main__":
    afs = AntiFerroSivak()
    print(afs.metric((1.9, 0.85)))