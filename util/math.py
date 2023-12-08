import numpy as np

def angle(v_i, v_j, radian=True):
    '''
calculate angle between two vectors 'v_i' and 'v_j' 
using numpy.arccos

Args:
    v_i (vector): Vector from origin
    v_j (vector): Vector from origin

Returns:
    Angle
    '''
    if len(np.array(v_i).shape) > 1: return angle_v(v_i, v_j, radian=radian)
    vi = np.array(v_i)/np.linalg.norm(v_i)
    vj = np.array(v_j)/np.linalg.norm(v_j)
    dot = np.dot(vi, vj)
    if dot > 1: dot = 1
    if dot < -1: dot = -1
    rad = np.arccos(dot)
    if radian: 
        return rad
    return rad*180.0/np.pi

def angle_v(v_i, v_j, radian=True, axis=-1):
    '''
calculate angle between two vectors 'v_i' and 'v_j' 
using numpy.arccos

Args:
    v_i (vector): Vector from origin
    v_j (vector): Vector from origin

Returns:
    Angle
    '''
    vi = np.array(v_i)/np.expand_dims(np.linalg.norm(v_i, axis=axis), axis)
    vj = np.array(v_j)/np.expand_dims(np.linalg.norm(v_j, axis=axis), axis)
    dot = np.expand_dims(np.sum(vi*vj, axis=axis), axis=axis)
    dot[dot >  1] = 1
    dot[dot < -1] = -1
    rad = np.arccos(dot)
    if radian: 
        return rad
    return rad*180.0/np.pi

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)