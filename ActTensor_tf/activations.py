import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from keras import backend as K

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

class SoftShrink(tf.keras.layers.Layer):
    def __init__(self, lamd=1.0, trainable=False, **kwargs):
        """
        Soft Shrinkage (Softshrink) Activation Layer
        
        Parameters
        ----------
        lamd : int, float
            lamd factor.
        """
        super(SoftShrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lamd = lamd
        self.trainable = trainable

    def build(self, input_shape):
        self.lambda_factor = K.variable(self.lamd,
                                      dtype=K.floatx(),
                                      name='lambda_factor')
        if self.trainable:
            self._trainable_weights.append(self.lambda_factor)

        super(SoftShrink, self).build(input_shape)

    def call(self, inputs, mask=None):
        return softSHRINK(inputs, self.lamd)
    
    def get_config(self):
        config = {'lambda': self.get_weights()[0] if self.trainable else self.lamd,
                  'trainable': self.trainable}
        base_config = super(SoftShrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class HardShrink(tf.keras.layers.Layer):
    def __init__(self, lamd=1.0, trainable=False, **kwargs):
        """
        Hard Shrinkage (Hardshrink) Activation Layer
        
        Parameters
        ----------
        lamd : int, float
            lamd factor.
        """
        super(HardShrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lamd = lamd
        self.trainable = trainable

    def build(self, input_shape):
        self.lambda_factor = K.variable(self.lamd,
                                      dtype=K.floatx(),
                                      name='lambda_factor')
        if self.trainable:
            self._trainable_weights.append(self.lambda_factor)

        super(HardShrink, self).build(input_shape)

    def call(self, inputs, mask=None):
        return hard_shrink(inputs, self.lamd)
    
    def get_config(self):
        config = {'lambda': self.get_weights()[0] if self.trainable else self.lamd,
                  'trainable': self.trainable}
        base_config = super(HardShrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class GLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        GLU Activation Layer
        """
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x


class Bilinear(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        Bilinear Activation Layer
        """
        super(Bilinear, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        x = tf.multiply(out, gate)
        return x


class ReGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        ReGLU Activation Layer
        """
        super(ReGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.nn.relu(gate)
        x = tf.multiply(out, gate)
        return x


class GeGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        GeGLU Activation Layer
        """
        super(GeGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.keras.activations.gelu(gate)
        x = tf.multiply(out, gate)
        return x


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        SwiGLU Activation Layer
        """
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.keras.activations.swish(gate)
        x = tf.multiply(out, gate)
        return x


class SeGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        SeGLU Activation Layer
        """
        super(SeGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.keras.activations.selu(gate)
        x = tf.multiply(out, gate)
        return x
    
class ReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Rectified Linear Unit Activation Layer
        """
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return relu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Linear Activation Layer f(x)=x
        """
        super(Identity, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Identity, self).build(input_shape)

    def call(self, inputs, mask=None):
        return identity(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Step(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Binary Step Activation Layer
        """
        super(Step, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Step, self).build(input_shape)

    def call(self, inputs, mask=None):
        return step(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Sigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Sigmoid Activation Layer
        """
        super(Sigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Sigmoid, self).build(input_shape)

    def call(self, inputs, mask=None):
        return sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Hard Sigmoid Activation Layer
        """
        super(HardSigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(HardSigmoid, self).build(input_shape)

    def call(self, inputs, mask=None):
        return hard_sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class LogSigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        LogSigmoid Activation Layer
        """
        super(LogSigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LogSigmoid, self).build(input_shape)

    def call(self, inputs, mask=None):
        return log_sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SiLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Sigmoid Linear Unit Activation Layer
        """
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(SiLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return silu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ParametricLinear(tf.keras.layers.Layer):
    def __init__(self, alpha=1., **kwargs):
        """
        Linear Activation Layer with parameter alpha
        
        Parameters
        ----------
        alpha : int, float default=1.0
        """
        super(ParametricLinear, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha

    def build(self, input_shape):
        super(ParametricLinear, self).build(input_shape)

    def call(self, inputs, mask=None):
        return parametric_linear(inputs, self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class PiecewiseLinear(tf.keras.layers.Layer):
    def __init__(self, xmin, xmax, **kwargs):
        """
        Piecewise Linear Activation Layer
        
        Parameters
        ----------
        xmin : int, float
            min range.
        xmax : int, float
            max range.
        """
        super(PiecewiseLinear, self).__init__(**kwargs)
        self.supports_masking = True
        self.xmin = xmin
        self.xmax = xmax

    def build(self, input_shape):
        super(PiecewiseLinear, self).build(input_shape)

    def call(self, inputs, mask=None):
        return piecewise_linear(inputs, self.xmin, self.xmax)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class CLL(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Complementary Log-Log Activation Layer
        """
        super(CLL, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(CLL, self).build(input_shape)

    def call(self, inputs, mask=None):
        return cll(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Bipolar(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Bipolar Activation Layer
        """
        super(Bipolar, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Bipolar, self).build(input_shape)

    def call(self, inputs, mask=None):
        return bipolar(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class BipolarSigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Bipolar Sigmoid Activation Layer
        """
        super(BipolarSigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(BipolarSigmoid, self).build(input_shape)

    def call(self, inputs, mask=None):
        return bipolar_sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Tanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Hyperbolic Tangent Activation Layer
        """
        super(Tanh, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Tanh, self).build(input_shape)

    def call(self, inputs, mask=None):
        return tanh(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class TanhShrink(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        TanhShrink Activation Layer
        """
        super(TanhShrink, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(TanhShrink, self).build(input_shape)

    def call(self, inputs, mask=None):
        return tanhshrink(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class LeCunTanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        LeCun's Tanh Activation Layer
        """
        super(LeCunTanh, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LeCunTanh, self).build(input_shape)

    def call(self, inputs, mask=None):
        return leCun_tanh(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class HardTanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Hard Tanh Activation Layer
        """
        super(HardTanh, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(HardTanh, self).build(input_shape)

    def call(self, inputs, mask=None):
        return hard_tanh(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class TanhExp(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Tanh Exponential Activation Layer
        """
        super(TanhExp, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(TanhExp, self).build(input_shape)

    def call(self, inputs, mask=None):
        return tanh_exp(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ABS(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Absolute Activation Layer
        """
        super(ABS, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ABS, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Abs(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SquaredReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Squared Rectified Linear Unit Activation Layer
        """
        super(SquaredReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(SquaredReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return squared_relu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ParametricReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.001, **kwargs):
        """
        Parametric Rectified Linear Unit Activation Layer
        
        Parameters
        ----------
        alpha : int, float default=0.001
        """
        super(ParametricReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha

    def build(self, input_shape):
        super(ParametricReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Parametric_ReLU(inputs, self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class RandomizedReLU(tf.keras.layers.Layer):
    def __init__(self, lower=0., upper=1., **kwargs):
        """
        Randomized Leaky Rectified Linear Unit Activation Layer
        
        Parameters
        ----------
        lower : int, float default=0
            lower range for random.uniform.
        upper : int, float default=1
            upper range for random.uniform.
        """
        super(RandomizedReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.lower = lower
        self.upper = upper

    def build(self, input_shape):
        super(RandomizedReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Randomized_ReLU(inputs, self.lower, self.upper)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Leaky Rectified Linear Unit Activation Layer
        """
        super(LeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(LeakyReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return leaky_ReLU(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ReLU6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        ReLU6 Activation Layer
        """
        super(ReLU6, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ReLU6, self).build(input_shape)

    def call(self, inputs, mask=None):
        return relu6(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ModReLU(tf.keras.layers.Layer):
    def __init__(self, bias, **kwargs):
        """
        Mod Rectified Linear Unit Activation Layer
        
        Parameters
        ----------
        bias : int, float
        """
        super(ModReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias

    def build(self, input_shape):
        super(ModReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Mod_ReLU(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class CosReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Cosine ReLU Activation Layer
        
        a = σ(z) = max(0, z) + cos(z)
        """
        super(CosReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(CosReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Cos_ReLU(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SinReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Sin ReLU Activation Layer
        
        a = σ(z) = max(0, z) + sin(z)
        """
        super(SinReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(SinReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Sin_ReLU(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Probit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Probit Activation Layer also known as Cumulative distribution function (CDF)
        """
        super(Probit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Probit, self).build(input_shape)

    def call(self, inputs, mask=None):
        return probit(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Cos(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Cos Activation Layer
        """
        super(Cos, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Cos, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Cosine(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Gaussian(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Gaussian Activation Layer
        """
        super(Gaussian, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Gaussian, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gaussian(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Multiquadratic(tf.keras.layers.Layer):
    def __init__(self, px, py, **kwargs):
        """
        Multiquadratic Activation Layer
        
        Parameters
        ----------
        px: float
            x dimension of chosen point
        py: float
            y dimension of chosen point
            
        notes
        -----
        px and py must be float otherwise it will get an error.
        """
        super(Multiquadratic, self).__init__(**kwargs)
        self.supports_masking = True
        self.px = px
        self.py = py

    def build(self, input_shape):
        super(Multiquadratic, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Multi_quadratic(inputs, self.px, self.py)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class InvMultiquadratic(tf.keras.layers.Layer):
    def __init__(self, px, py, **kwargs):
        """
        Inverse Multiquadratic Activation Layer
        
        Parameters
        ----------
        px: float
            x dimension of chosen point
        py: float
            y dimension of chosen point
            
        notes
        -----
        px and py must be float otherwise it will get an error.
        """
        super(InvMultiquadratic, self).__init__(**kwargs)
        self.supports_masking = True
        self.px = px
        self.py = py

    def build(self, input_shape):
        super(InvMultiquadratic, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Inv_Multi_quadratic(inputs, self.px, self.py)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SoftPlus(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Softplus or Smooth ReLU Activation Layer
        """
        super(SoftPlus, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(SoftPlus, self).build(input_shape)

    def call(self, inputs, mask=None):
        return softPlus(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Mish Activation Layer
        """
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return mish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Smish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Mish Activation Layer
        """
        super(Smish, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Smish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return smish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ParametricSmish(tf.keras.layers.Layer):
    def __init__(self, alpha = 1., beta = 1., **kwargs):
        """
        Parametric Smish Activation Layer
        
        Parameters
        ----------
        alpha : float, default=1.
                alpha weight.
        beta : float, default=1.
                beta weight.
            
        notes
        -----
        alpha and beta must be float otherwise it will get an error.
        """
        super(ParametricSmish, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.beta = beta

    def build(self, input_shape):
        super(ParametricSmish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Parametric_Smish(inputs, self.alpha, self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Swish(tf.keras.layers.Layer):
    def __init__(self, beta = 1., **kwargs):
        """
        Swish Activation Layer
        
        Parameters
        ----------
        beta : int, float default=1.
        """
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta

    def build(self, input_shape):
        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ESwish(tf.keras.layers.Layer):
    def __init__(self, beta = 1., **kwargs):
        """
        E-Swish Activation Layer
        
        Parameters
        ----------
        beta : int, float default=1.
        """
        super(ESwish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta

    def build(self, input_shape):
        super(ESwish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return eswish(inputs, self.beta)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class HardSwish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Hard Swish Activation Layer
        """
        super(HardSwish, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(HardSwish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return hardSwish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class GCU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Growing Cosine Unit Activation Layer
        """
        super(GCU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(GCU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gcu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class CoLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Collapsing Linear Unit Activation Layer
        """
        super(CoLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(CoLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return colu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class PELU(tf.keras.layers.Layer):
    def __init__(self, c, b, alpha, **kwargs):
        """
        Parametric Exponential Linear Unit Activation Layer
        
        Parameters
        ----------
        alpha : int, float
        c : int, float
        b : int, float
        """
        super(PELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.b = b
        self.c = c

    def build(self, input_shape):
        super(PELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return pelu(inputs, self.c, self.b,self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        SELU Activation Layer
        """
        super(SELU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(SELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return selu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class CELU(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        """
        CELU Activation Layer
        
        Parameters
        ----------
        alpha : int, float, default=1.0
        """
        super(CELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha

    def build(self, input_shape):
        super(CELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return celu(inputs, self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ArcTan(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        ArcTang Activation Layer
        """
        super(ArcTan, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ArcTan, self).build(input_shape)

    def call(self, inputs, mask=None):
        return arcTan(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ShiftedSoftPlus(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Shifted Softplus Activation Layer
        """
        super(ShiftedSoftPlus, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ShiftedSoftPlus, self).build(input_shape)

    def call(self, inputs, mask=None):
        return Shifted_SoftPlus(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Softmax(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Softmax Activation Layer
        """
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Softmax, self).build(input_shape)

    def call(self, inputs, mask=None):
        return softmax(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Logit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Logit Activation Layer
        """
        super(Logit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Logit, self).build(input_shape)

    def call(self, inputs, mask=None):
        return logit(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Gaussian Error Linear Unit Activation Layer
        """
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Softsign(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Softsign Activation Layer
        """
        super(Softsign, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Softsign, self).build(input_shape)

    def call(self, inputs, mask=None):
        return softsign(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ELiSH(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Exponential Linear Squashing Activation Layer
        """
        super(ELiSH, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(ELiSH, self).build(input_shape)

    def call(self, inputs, mask=None):
        return elish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class HardELiSH(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Hard Exponential Linear Squashing Activation Layer
        """
        super(HardELiSH, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(HardELiSH, self).build(input_shape)

    def call(self, inputs, mask=None):
        return hardELiSH(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Serf(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Log-Softplus Error Activation Layer
        """
        super(Serf, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Serf, self).build(input_shape)

    def call(self, inputs, mask=None):
        return serf(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ELU(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        """
        Exponential Linear Unit Activation Layer
        
        Parameters
        ----------
        alpha : int,float
        """
        super(ELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha

    def build(self, input_shape):
        super(ELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return elu(inputs, self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Phish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Phish Activation Layer
        """
        super(Phish, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Phish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return phish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class QReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Quantum Rectifier Linear Unit Activation Layer
        """
        super(QReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(QReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return qrelu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class MQReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        modified QReLU Activation Layer
        """
        super(MQReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(MQReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return mqrelu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class FReLU(tf.keras.layers.Layer):
    def __init__(self, b, **kwargs):
        """
        Flexible Rectified Linear Unit Activation Layer
        
        Parameters
        ----------
        b : int, float
        """
        super(FReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.b = b

    def build(self, input_shape):
        super(FReLU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return frelu(inputs, self.b)

    def compute_output_shape(self, input_shape):
        return input_shape