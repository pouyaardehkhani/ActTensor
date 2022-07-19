from act_tensor.functions import *
import tensorflow as tf
import numpy as np
from keras import backend
from keras.backend import *
from tensorflow.keras import backend as K

class SoftShrink(tf.keras.layers.Layer):
    def __init__(self, lamd=1.0, trainable=False, **kwargs):
        """
        Soft Shrinkage (Softshrink) Activation Layer
        
        Parameters
        ----------
        lamd : int, float
        trainable : default=False
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
        return softSHRINK(inputs, self.lambda_factor)
    
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
        trainable : default=False
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
        return hard_shrink(inputs, self.lambda_factor)
    
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