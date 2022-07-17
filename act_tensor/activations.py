import tensorflow as tf
import numpy as np

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
    return 1 if x > 0 else 0