import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from tensorflow.keras import backend as K

def hard_shrink(x, lamd):
    """
    Hard Shrinkage (Hardshrink) Activation Function
    
    Parameters
    ----------
    x : tensor object
    lamd : int, float
    
    Returns
    -------
    tensor
    """
    x = tf.where((-lamd<=x) & (x<=lamd), x, 0)
    return x

def relu(x):
    """
    Rectified Linear Unit Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    x = tf.where(x>0, x, 0)
    return x

def identity(x):
    """
    Linear Activation Function f(x)=x
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x

def step(x):
    """
    Binary Step Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    x = tf.where(x>0,1,0)
    return x

def sigmoid(x):
    """
    Sigmoid Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return 1/(1 + tf.math.exp(-x))

def hard_sigmoid(x):
    """
    Hard Sigmoid Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0., tf.math.minimum(1., (x + 1.)/2.))

def log_sigmoid(x):
    """
    LogSigmoid Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.log(sigmoid(x))

def silu(x):
    """
    Sigmoid Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x / (1. - tf.math.exp(-x))

def parametric_linear(x, a):
    """
    Linear Activation Function with parameter a
        
    Parameters
    ----------
    x : tensor object
    a : int, float
        alpha weight for x.
        
    Returns
    -------
    tensor
    """
    return a * x

def piecewise_linear(x, xmin, xmax):
    """
    Piecewise Linear Activation Function
    
    Choose some xmin and xmax, which is our "range". Everything less than than this range will be 0, and everything greater than this range will be 1. Anything else is linearly-interpolated between.
        
    Parameters
    ----------
    x : tensor object
    xmin : int, float
        min range.
    xmax : int, float
        max range.
        
    Returns
    -------
    tensor
    """
    m = 1./(xmax-xmin)
    b = 1. - (m * xmax)
    x = tf.where((x >= xmin) & (x <= xmax), tf.add(tf.multiply(x,m),b),x)
    x = tf.where(x > 1., 1.,x)
    x = tf.where(x < 0., 0.,x)
    return x

def cll(x):
    """
    Complementary Log-Log Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return 1. - (tf.math.exp(-(tf.math.exp(x))))

def bipolar(x):
    """
    Bipolar Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    x = tf.where(x > 0., 1., -1.)
    return x

def bipolar_sigmoid(x):
    """
    Bipolar Sigmoid Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (1. - tf.math.exp(-x))/(1. + tf.math.exp(-x))

def tanh(x):
    """
    Hyperbolic Tangent Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (2./(1. + tf.math.exp(-2.*x))) - 1.

def tanhshrink(x):
    """
    TanhShrink Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x - tanh(x)

def leCun_tanh(x):
    """
    LeCun's Tanh Activation Function
    
    Used for efficient backprop.
    
    Output Range : (-1.7159 to 1.7159)
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (1.7159 * tf.math.tanh((2./3.) * x))

def hard_tanh(x):
    """
    Hard Tanh Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(-1., tf.math.minimum(1., x))

def tanh_exp(x):
    """
    Tanh Exponential Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x * tanh(tf.math.exp(x))

def Abs(x):
    """
    Absolute Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.abs(x)

def squared_relu(x):
    """
    Squared Rectified Linear Unit Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    x = tf.where(x > 0., tf.math.pow(x, 2.), 0.)
    return x

def Parametric_ReLU(x, alpha):
    """
    Parametric Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int, float
        
    Returns
    -------
    tensor
    """
    x = tf.where(x>0., x, alpha*x)
    return x

def Randomized_ReLU(x, lower, upper):
    """
    Randomized Leaky Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    lower : int, float
        lower range for random.uniform.
    upper : int, float
        upper range for random.uniform.
        
    Returns
    -------
    tensor
    """
    a = np.random.uniform(lower, upper, 1)
    x = tf.where(x>=0., x, a*x)
    return x

def leaky_ReLU(x):
    """
    Leaky Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    x = tf.where(x>0., x, 0.01*x)
    return x

def relu6(x):
    """
    ReLU6 Activation Function
       
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.minimum(tf.math.maximum(0., x), 6.)

def Mod_ReLU(x, bias):
    """
    Mod Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    bias : int, float
        
    Returns
    -------
    tensor
    """
    x = tf.where(tf.abs(x)+bias>=0., (tf.abs(x)+bias)* (x/tf.abs(x)), 0.)
    return x

def Cos_ReLU(x):
    """
    Cosine ReLU Activation Function 
    
    a = σ(z) = max(0, z) + cos(z)
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0.,x) + tf.math.cos(x)

def Sin_ReLU(x):
    """
    Sin ReLU Activation Function
    
    a = σ(z) = max(0, z) + sin(z)
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0.,x) + tf.math.sin(x)

def probit(x):
    """
    Probit Activation Function also known as Cumulative distribution function (CDF)
        
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
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.cos(x)

def gaussian(x):
    """
    Gaussian Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.exp(tf.math.multiply(-0.5, tf.math.pow(x, 2.)))

def Multi_quadratic(x, px, py):
    """
    Multiquadratic Activation Function
    
    Parameters
    ----------
    x : tensor object
    px: int, float
        x dimension of chosen point
    py: int, float
        y dimension of chosen point
        
    Returns
    -------
    tensor
    
    notes
    -----
    px and py must be float otherwise it will get an error.
    """
    return tf.math.sqrt(tf.math.add(tf.math.pow(tf.math.subtract(x,px ),2.), tf.math.pow(py, 2.)))

def Inv_Multi_quadratic(x, px, py):
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
    px and py must be float otherwise it will get an error.
    """
    return 1./(tf.math.sqrt(tf.math.add(tf.math.pow(tf.math.subtract(x,px ),2.), tf.math.pow(py, 2.))))

def softPlus(x):
    """
    Softplus or Smooth ReLU Activation Function
    
    Output Range : (0, infinity)
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.math.log(1. + tf.math.exp(x))

def mish(x):
    """
    Mish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x * tanh(softPlus(x))

def smish(x):
    """
    Smish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.multiply(x, tf.math.tanh(tf.math.log(tf.math.add(1., sigmoid(x)))))

def Parametric_Smish(x, alpha = 1., beta = 1.):
    """
    Parametric Smish Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : float, default=1.
            alpha weight.
    beta : float, default=1.
            beta weight.
        
    Returns
    -------
    tensor
    
    notes
    -----
    alpha and beta must be float otherwise it will get an error.
    """
    a = tf.math.multiply(alpha, x)
    b = tf.math.multiply(beta, x)
    return tf.math.multiply(a, tf.math.tanh(tf.math.log(tf.math.add(1., sigmoid(b)))))

def swish(x, beta):
    """
    Swish Activation Function
    
    Parameters
    ----------
    x : tensor object
    beta : int, float
        
    Returns
    -------
    tensor
    """
    return x / (1. - tf.math.exp(-beta*x))

def eswish(x, beta):
    """
    E-Swish Activation Function
    
    Parameters
    ----------
    x : tensor object
    beta : int, float
        
    Returns
    -------
    tensor
    """
    return beta * (x / (1. - tf.math.exp(-beta*x)))

def hardSwish(x):
    """
    Hard Swish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x * (relu6(x+3) / 6)

def gcu(x):
    """
    Growing Cosine Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.multiply(x, tf.math.cos(x))

def colu(x):
    """
    Collapsing Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return  x / (1. - x*(tf.math.exp(-x-tf.math.exp(x))))

def softSHRINK(x, lambd):
    """
    SOFTSHRINK Activation Function
    
    Parameters
    ----------
    x : tensor object
    lambd : int, float
        
    Returns
    -------
    tensor
    """
    x = tf.where((x > (-lambd)) & (x < lambd),0.,x)
    x = tf.where(x >= lambd, x - lambd, x)
    x = tf.where(x <= (-lambd), x + lambd, x)
    return x

def pelu(x, c, b, alpha):
    """
    Parametric Exponential Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int, float
    c : int, float
    b : int, float
    
    Returns
    -------
    tensor
    """
    x = tf.where(x>0., c*x, alpha*(tf.math.exp(x/b)-1.))
    return x

def selu(x):
    """
    SELU Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * (tf.math.maximum(0.,x) + tf.math.minimum(0.,alpha*(tf.math.exp(x)-1.)))

def celu(x, alpha=1.0):
    """
    CELU Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int, float, default=1.0
    
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0.,x) + tf.math.minimum(0.,alpha*(tf.math.exp(x/alpha)-1.))

def arcTan(x):
    """
    ArcTang Activation Function
        
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.math.atan(x)

def Shifted_SoftPlus(x):
    """
    Shifted Softplus Activation Function
        
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.math.log(0.5 + 0.5*tf.math.exp(x))

def softmax(x):
    """
    Softmax Activation Function
        
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.keras.activations.softmax(x, axis=-1)

def logit(x):
    """
    Logit Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x / (1.-x)

def gelu(x):
    """
    Gaussian Error Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.keras.activations.gelu(x)

def softsign(x):
    """
    Softsign Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x / (tf.math.abs(x) + 1.)

def elish(x):
    """
    Exponential Linear Squashing Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    x = tf.where(x>=0., x/(1.+tf.math.exp(-x)), ((tf.math.exp(x)-1.)/(1.+tf.math.exp(-x))))
    return x

def hardELiSH(x):
    """
    Hard Exponential Linear Squashing Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    x = tf.where(x>=0., x*tf.math.maximum(0., tf.math.minimum(1., (x+1.)/2.)), (tf.math.exp(x)-1.)*tf.math.maximum(0., tf.math.minimum(1., (x+1.)/2.)))
    return x

def serf(x):
    """
    Log-Softplus Error Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x * tf.math.erf(tf.math.log(1.+tf.math.exp(x)))

def elu(x, alpha):
    """
    Exponential Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int,float
    
    Returns
    -------
    tensor
    """
    x = tf.where(x>0., x, alpha*tf.math.exp(x)-1.)
    return x

def phish(x):
    """
    Phish Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x * tanh(gelu(x))

def qrelu(x):
    """
    Quantum Rectifier Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.where(x>0.,x,(0.01*(x-2))*x)

def mqrelu(x):
    """
    modified QReLU Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return tf.where(x>0.,x,(0.01*(x)) - x)

def frelu(x, b):
    """
    Flexible Rectified Linear Unit (FReLU) Activation Function
    
    Parameters
    ----------
    x : tensor object
    b : int, float
    
    Returns
    -------
    tensor
    """
    return tf.where(x>0.,x+b,b)