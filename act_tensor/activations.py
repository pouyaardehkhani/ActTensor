import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from tensorflow.keras import backend as K

def Identity(x):
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

def Step(x):
    """
    Binary Step Activation Function
        
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

def Sigmoid(x):
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

def HardSigmoid(x):
    """
    Hard Sigmoid Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0, tf.math.minimum(1, (x+1)/2))

def LogSigmoid(x):
    """
    LogSigmoid Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.log(Sigmoid(x))

def SiLU(x):
    """
    Sigmoid Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x / (1 - tf.math.exp(-x))

def Linear(x, a):
    """
    Linear Activation Function
        
    Parameters
    ----------
    x : tensor object
    a : int, float
        
    Returns
    -------
    tensor
    """
    return a * x

def PLinear(x, xmin, xmax):
    """
    Piecewise Linear Activation Function
    
    Choose some xmin and xmax, which is our "range". Everything less than than this range will be 0, and everything greater than this range will be 1. Anything else is linearly-interpolated between.
        
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

def BipolarSigmoid(x):
    """
    Bipolar Sigmoid Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return (1-tf.math.exp(-x))/(1+tf.math.exp(-x))

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
    return (2/(1 + tf.math.exp(-2*x))) - 1

def tanhShrink(x):
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

def LeCunTanh(x):
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
    return (1.7159 * tf.math.tanh((2/3) * x))

def HardTanh(x):
    """
    Hard Tanh Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.maximum(-1, tf.math.minimum(1, x))

def TanhExp(x):
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

def ReLU(x):
    """
    Rectified Linear Unit Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>0, x, 0)
    x = tf.cast(x, dtype)
    return x

def SquaredReLU(x):
    """
    Squared Rectified Linear Unit Activation Function
        
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>0, x**2, 0)
    x = tf.cast(x, dtype)
    return x

def PReLU(x, alpha):
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
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>0, x, alpha*x)
    x = tf.cast(x, dtype)
    return x

def RReLU(x, lower, upper):
    """
    Randomized Leaky Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    a = np.random.uniform(lower, upper, 1)
    x = np.where(x>=0, x, a*x)
    x = tf.cast(x, dtype)
    return x

def LeakyReLU(x):
    """
    Parametric Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>0, x, 0.01*x)
    x = tf.cast(x, dtype)
    return x

def ReLU6(x):
    """
    Mish Activation Function
       
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.minimum(tf.math.maximum(0, x), 6)

def ModReLU(x, bias):
    """
    Parametric Rectified Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
    bias : int, float
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(np.abs(x)+bias>=0, (np.abs(x)+bias)* (x/np.abs(x)), 0)
    x = tf.cast(x, dtype)
    return x

def CosR(x):
    """
    Modification for ReLU Activation Function 
    
    a = σ(z) = max(0, z) + cos(z).
        
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

def Mish(x):
    """
    Mish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x * tanh(SoftPlus(x))

def Smish(x):
    """
    Smish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return tf.math.multiply(x, tf.math.tanh(tf.math.log(tf.math.add(1., Sigmoid(x)))))

def PSmish(x, alpha = 1., beta = 1.):
    """
    Parametric Smish Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : float, default 1.
            alpha weight.
    alpha : float, default 1.
            beta weight.
        
    Returns
    -------
    tensor
    """
    a = tf.math.multiply(alpha, x)
    b = tf.math.multiply(beta, x)
    return tf.math.multiply(a, tf.math.tanh(tf.math.log(tf.math.add(1., Sigmoid(b)))))

def Swish(x, beta):
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
    return x / (1 - tf.math.exp(-beta*x))

def ESwish(x, beta):
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
    return beta * (x / (1 - tf.math.exp(-beta*x)))

def HardSwish(x):
    """
    Hard Swish Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    return x * (ReLU6(x+3) / 6)

def GCU(x):
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

def CoLU(x):
    """
    Collapsing Linear Unit Activation Function
    
    Parameters
    ----------
    x : tensor object
        
    Returns
    -------
    tensor
    """
    b = tf.math.multiply(-1.,(tf.math.add(x, tf.math.exp(x))))
    a = tf.math.pow(x,b)
    return tf.math.divide(x,tf.math.subtract(1., a))

def SOFTSHRINK(x, alpha):
    """
    SOFTSHRINK Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int, float
        
    Returns
    -------
    tensor
    """
    dtype = getattr(x, "dtype", floatx())
    x.numpy()
    x = np.where((x > (-alpha)) & (x < alpha),0,x)
    x = np.where(x >= alpha, x - alpha, x)
    x = np.where(x <= (-alpha), x + alpha, x)
    x = tf.cast(x, dtype)
    return x

def PELU(x, c, b, alpha):
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
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>0, c*x, alpha*(np.exp(x/b)-1))
    x = tf.cast(x, dtype)
    return x

def SELU(x):
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
    return scale * (tf.math.maximum(0,x) + tf.math.minimum(0,alpha*(tf.math.exp(x)-1)))

def CELU(x, alpha=1.0):
    """
    SELU Activation Function
    
    Parameters
    ----------
    x : tensor object
    alpha : int, float, default=1.0
    
    Returns
    -------
    tensor
    """
    return tf.math.maximum(0,x) + tf.math.minimum(0,alpha*(tf.math.exp(x/alpha)-1))

def ArcTan(x):
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

def SoftPlus(x):
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
    return tf.math.log(1 + tf.math.exp(x))

def ShiftedSoftPlus(x):
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

def Softmax(x):
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

def Logit(x):
    """
    Logit Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x / (1-x)

def GELU(x):
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

def Softsign(x):
    """
    Softsign Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x / (tf.math.abs(x) + 1)

def ELiSH(x):
    """
    Exponential Linear Squashing Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>=0, x/(1+tf.math.exp(-x)), ((tf.math.exp(x)-1)/(1+tf.math.exp(-x))))
    x = tf.cast(x, dtype)
    return x

def HardELiSH(x):
    """
    Hard Exponential Linear Squashing Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where(x>=0, x*tf.math.maximum(0, tf.math.minimum(1, (x+1)/2)), (tf.math.exp(x)-1)*tf.math.maximum(0, tf.math.minimum(1, (x+1)/2)))
    x = tf.cast(x, dtype)
    return x

def Serf(x):
    """
    Log-Softplus Error Activation Function
    
    Parameters
    ----------
    x : tensor object
    
    Returns
    -------
    tensor
    """
    return x * tf.math.erf(tf.math.log(1+tf.math.exp(x)))

def HardShrink(x, lamd):
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
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where((-lamd<=x) & (x<=lamd), x, 0)
    x = tf.cast(x, dtype)
    return x

def SoftShrink(x, lamd):
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
    dtype = getattr(x, 'dtype', floatx())
    x.numpy()
    x = np.where((-lamd<=x) & (x<=lamd), x, 0)
    x = tf.cast(x, dtype)
    return x
