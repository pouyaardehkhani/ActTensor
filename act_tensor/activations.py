import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from tensorflow.keras import backend as K

def Identity(x):
    """
    Linear activation function f(x)=x.
    
    Range : (-infinity to infinity)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x

def Step(x):
    """
    Binary step activation function.
    
    Range : (0 , 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, "dtype", floatx())
    x.numpy()
    x = np.where(x>0,1,0)
    x = tf.cast(x, dtype)
    return x

def PLinear(x, xmin, xmax):
    """
    Piecewise Linear activation function.
    
    Choose some xmin and xmax, which is our "range". Everything less than than this range will be 0, and everything greater than this range will be 1. Anything else is linearly-interpolated between.
    
    Range : (0 to 1)
    
    Parameters
    ----------
    x : tensor object
    xmin : int
        min range.
    xmax : int
        max range.
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, "dtype", floatx())
    x.numpy()
    m = 1./(xmax-xmin)
    b = 1 - (m * xmax)
    x = np.where((x >= xmin) & (x <= xmax), np.add(np.multiply(x,m),b),x)
    x = np.where(x > 1, 1,x)
    x = np.where(x < 0, 0,x)
    x = tf.cast(x, dtype)
    return x






    
    
    
    
