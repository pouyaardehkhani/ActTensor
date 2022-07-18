import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *

def Identity(x):
    """
    Linear activation function f(x)=x.
    
    Range : (-infinity to infinity)
    """
    return x

def Step(x):
    """
    Binary step activation function.
    
    Range : (0 , 1)
    """
    dtype = getattr(x, "dtype", floatx())
    x.numpy()
    x=np.where(x>0,1,0)
    x=tf.cast(x, dtype)
    return x


