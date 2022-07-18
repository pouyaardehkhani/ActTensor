import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from tensorflow.keras import backend as K

def Identity(x):
    """
    Linear Activation Function f(x)=x
    
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
    Binary Step Activation Function
    
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
    Piecewise Linear Activation Function
    
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

def Cll(x):
    """
    Complementary Log-Log Activation Function
    
    Range : (0 to 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return 1 - (tf.math.exp(-(tf.math.exp(x))))

def Bipolar(x):
    """
    Bipolar Activation Function
    
    Range : (-1 , 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, "dtype", floatx())
    x.numpy()
    x = np.where(x>0,1,-1)
    x = tf.cast(x, dtype)
    return x

def BSigmoid(x):
    """
    Bipolar Sigmoid Activation Function
    
    Range : (-1 to 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (1-tf.math.exp(-x))/(1+tf.math.exp(-x))

def LeCunTanh(x):
    """
    LeCun's Tanh Activation Function
    
    Used for efficient backprop.
    
    Range : (-1.7159 to 1.7159)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (1.7159 * tf.math.tanh((2/3) * x))

def HardTanh(x):
    """
    Hard Tanh Activation Function
    
    Range : (-1 to 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(-1, tf.math.minimum(1, x))

def Abs(x):
    """
    Absolute Activation Function
    
    Range : (0 to infinity)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.abs(x)

def CosReLU(x):
    """
    Modification for ReLU Activation Function 
    
    a = σ(z) = max(0, z) + cos(z).
    
    Range : (-1 to infinity)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0,x) + tf.math.cos(x)

def SinReLU(x):
    """
    Modification for ReLU Activation Function
    
    a = σ(z) = max(0, z) + sin(z).
    
    Range : (-1 to infinity)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0,x) + tf.math.sin(x)

def Probit(x):
    """
    Probit Activation Function also known as  Cumulative distribution function (CDF)
    
    Range : (-infinity to infinity)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.multiply(tf.math.sqrt(2.) , tf.math.erfinv( tf.math.subtract(tf.math.multiply(x, 2), 1)))

def Cosine(x):
    """
    Cosine Activation Function
    
    Range : (-1 to 1)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.cos(x)

def Gaussian(x):
    """
    Gaussian Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.exp(tf.math.multiply(-0.5, tf.math.pow(x, 2)))

def Multiquadratic(x, px, py):
    """
    Multiquadratic Activation Function
    
    Parameters
    ----------
    x : tensor object
    px: float
        x dimension of chosen point
    py: float
        y dimension of chosen point
        
    Returns
    -------
    tensor
    
    notes
    -----
    px and py must be float or this will not work.
    """
    return tf.math.sqrt(tf.math.add(tf.math.pow(tf.math.subtract(x,px ),2), tf.math.pow(py, 2)))

def InvMultiquadratic(x, px, py):
    """
    Inverse Multiquadratic Activation Function
    
    Parameters
    ----------
    x : tensor object
    px: float
        x dimension of chosen point
    py: float
        y dimension of chosen point
        
    Returns
    -------
    tensor
    
    notes
    -----
    px and py must be float or this will not work.
    """
    return 1./(tf.math.sqrt(tf.math.add(tf.math.pow(tf.math.subtract(x,px ),2), tf.math.pow(py, 2))))





    


    
    
