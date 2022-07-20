<div align="center">
  <img src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ActTensor%20logo.png"><br>
</div>

---------

# **ActTensor**: Activation Functions for TensorFlow

![license](https://img.shields.io/github/license/pouyaardehkhani/ActTensor.svg) ![releases](https://img.shields.io/github/release/pouyaardehkhani/ActTensor.svg)

## **What is it?**

ActTensor is a Python package that provides state-of-the-art activation functions which facilitate using them in Deep Learning projects in an easy and fast manner. 

## **Why not using tf.keras.activations?**
As you may know, TensorFlow only has a few defined activation functions and most importantly it does not include newly-introduced activation functions. Wrting another one requires time and energy; however, this package has most of the widely-used, and even state-of-the-art activation functions that are ready to use in your models.

## Requirements

    numpy
    tensorflow
    setuptools
    keras
    wheel
    
## Where to get it?
The source code is currently hosted on GitHub at:
https://github.com/pouyaardehkhani/ActTensor

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/ActTensor-tf/)

```sh
# PyPI
pip install ActTensor-tf
```

## License
[MIT](LICENSE)

## How to use?

```sh
import tensorflow as tf
import numpy as np
from ActTensor_tf import ReLU # name of the layer
```
functional api

```sh
inputs = tf.keras.layers.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128)(x)
# wanted class name
x = ReLU()(x)
output = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.models.Model(inputs = inputs,outputs=output)
```
sequential api 
```sh
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128),
                                    # wanted class name
                                    ReLU(),
                                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
```

NOTE:
> The main function of the activation layers are also availabe but it maybe defined as different name. Check [this](https://github.com/pouyaardehkhani/ActTensor/edit/master/README.md#activations) for more information.
```
from ActTensor_tf import relu
```

## Activations

Classes and Functions are available in ***ActTensor_tf***


| Activation Name | Class Name | Function Name |
| :---:        |     :---:      |         :---: |
| SoftShrink   | [SoftShrink](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L8)    | [softSHRINK](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L649)    |
| HardShrink     | [HardShrink](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L45)     | [hard_shrink](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L7)      |
| GLU     | [GLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L82)       | -      |
| Bilinear     | [Bilinear](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L99)       | -      |
| ReGLU     | [ReGLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L115)       | -      |
| GeGLU     | [GeGLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L132)       | -      |
| SwiGLU     | [SwiGLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L149)       | -      |
| SeGLU     | [SeGLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L166)       | -      |
| ReLU     | [ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L182)       | [relu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L23)      |
| Identity     | [Identity](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L199)       | [identity](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L38)      |
| Step     | [Step](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L216)       | [step](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L52)      |
| Sigmoid     | [Sigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L233)       | [sigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L67)     |
| HardSigmoid     | [HardSigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L250)       | [hard_sigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L81)      |
| LogSigmoid     | [LogSigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L267)       | [log_sigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L95)      |
|  SiLU     | [SiLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L284)       | [silu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L109)      |
| PLinear     | [ParametricLinear](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L301)       | [parametric_linear](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L123)      |
| Piecewise-Linear     | [PiecewiseLinear](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L323)       | [piecewise_linear](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L139)     |
| Complementary Log-Log     | [CLL](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L349)       | [cll](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L164)      |
| Bipolar     | [Bipolar](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L366)       | [bipolar](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L178)     |
| Bipolar-Sigmoid     | [BipolarSigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L383)       | [bipolar_sigmoid](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L193)      |
| Tanh     | [Tanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L400)       | [tanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L207)      |
| TanhShrink     | [TanhShrink](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L417)       | [tanhshrink](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L221)      |
| LeCun's Tanh     | [LeCunTanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L434)      | [leCun_tanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L235)      |
| HardTanh     | [HardTanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L451)       | [hard_tanh](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L253)      |
| TanhExp     | [TanhExp](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L468)       | [tanh_exp](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L267)      |
| Absolute     | [ABS](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L485)       | [Abs](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L281)      |
| Squared-ReLU     | [SquaredReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L502)       | [squared_relu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L295)      |
| P-ReLU     | [ParametricReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L519)      | [Parametric_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L310)     |
| R-ReLU     | [RandomizedReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L541)      | [Randomized_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L326)      |
| LeakyReLU     | [LeakyReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L567)       | [leaky_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L346)     |
| ReLU6     | [ReLU6](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L584)       | [relu6](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L361)      |
| Mod-ReLU     | [ModReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L601)       | [Mod_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L375)      |
| Cosine-ReLU     | [CosReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L623)       | [Cos_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L391)      |
| Sin-ReLU     | [SinReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L642)       | [Sin_ReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L407)      |
| Probit     | [Probit](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L661)       | [probit](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L423)      |
| Cos     | [Cos](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L678)      | [Cosine](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L437)      |
| Gaussian     | [Gaussian](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L695)       | [gaussian](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L451)      |
| Multiquadratic     | [Multiquadratic](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L712)       | [Multi_quadratic](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L465)      |
| Inverse-Multiquadratic     | [InvMultiquadratic](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L742)       | [Inv_Multi_quadratic](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L487)      |
| SoftPlus     | [SoftPlus](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L772)       | [softPlus](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L509)      |
| Mish     | [Mish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L789)      | [mish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L525)      |
| SMish     | [Smish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L806)       | [smish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L539)      |
| P-SMish     | [ParametricSmish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L823)       | [Parametric_Smish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L553)      |
| Swish     | [Swish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L853)      | [swish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L577)      |
| ESwish     | [ESwish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L875)      | [eswish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L592)      |
| HardSwish     | [HardSwish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L897)       | [hardSwish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L607)      |
| GCU     | [GCU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L914)       | [gcu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L621)      |
| CoLU     | [CoLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L931)       | [colu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L635)      |
| PELU     | [PELU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L948)       | [pelu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L667)      |
| SELU     | [SELU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L974)       | [selu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L685)      |
| CELU     | [CELU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L991)       | [celu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L701)      |
| ArcTan     | [ArcTan](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1013)       | [arcTan](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L716)      |
| Shifted-SoftPlus     | [ShiftedSoftPlus](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1030)       | [Shifted_SoftPlus](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L730)      |
| Softmax     | [Softmax](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1047)       | [softmax](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L744)      |
| Logit     | [Logit](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1064)       | [logit](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L758)      |
| GELU     | [GELU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1081)       | [gelu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L772)      |
| Softsign     | [Softsign](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1098)       | [softsign](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L786)      |
| ELiSH     | [ELiSH](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1115)       | [elish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L800)      |
| HardELiSH     | [HardELiSH](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1132)       | [hardELiSH](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L815)      |
| Serf     | [Serf](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1149)       | [serf](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L830)      |
| ELU     | [ELU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1166)       | [elu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L844)      |
| Phish     | [Phish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1188)       | [phish](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L860)      |
| QReLU     | [QReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1205)       | [qrelu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L874)      |
| MQReLU     | [MQReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1222)       | [mqrelu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L888)      |
| FReLU     | [FReLU](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/layers.py#L1239)       | [frelu](https://github.com/pouyaardehkhani/ActTensor/blob/fd5adadc18b9cf9a060d43e48d3ede7057ff11d3/act_tensor/functions.py#L902)      |


<div align="center">
  <img src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Activation%20Functions.gif"><br>
</div>



## **Which activation functions it supports?**

1. Soft Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x-\lambda&space;&&space;x&space;>&space;\lambda\\&space;x&plus;\lambda&space;&&space;x&space;<&space;-\lambda\\&space;0&space;&&space;otherwise&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SoftShrink.png"> 
</p>

2. Hard Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\begin{cases}x&space;&&space;x&space;>&space;\lambda\\&space;x&space;&&space;x&space;<&space;-\lambda\\&space;0&space;&&space;otherwise&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/HardShrink.png"> 
</p>

3. GLU:

    <img src="https://latex.codecogs.com/svg.image?&space;GLU\left&space;(&space;a,b&space;&space;\right&space;)=&space;a&space;\oplus&space;\sigma&space;\left&space;(&space;b&space;\right&space;)">

<p align="center"> 
  <img width="700" height="500" src="https://production-media.paperswithcode.com/methods/new_architecture_8UYjVkL.jpg"> 
</p>

* [Source Paper : Language Modeling with Gated Convolutional Networks](http://arxiv.org/abs/1612.08083v3)

4. Bilinear:

* [Source Paper : Parameter Efficient Deep Neural Networks with Bilinear Projections](https://arxiv.org/pdf/2011.01391)

5. ReGLU:

    ReGLU is an activation function which is a variant of GLU. 

    <img src="https://latex.codecogs.com/svg.image?ReGLU\left&space;(&space;x,&space;W,&space;V,&space;b,&space;c&space;\right&space;)=&space;max(0,&space;xW&space;&plus;&space;b)&space;\oplus&space;(xV&space;&plus;&space;b)">

* [Source Paper : GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)

6. GeGLU:

    GeGLU is an activation function which is a variant of GLU. 

    <img src="https://latex.codecogs.com/svg.image?GeGLU\left&space;(&space;x,&space;W,&space;V,&space;b,&space;c&space;\right&space;)=&space;GELU(xW&space;&plus;&space;b)&space;\oplus&space;(xV&space;&plus;&space;b)">

* [Source Paper : GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)

7. SwiGLU:

    SwiGLU is an activation function which is a variant of GLU. 

    <img src="https://latex.codecogs.com/svg.image?SwiGLU\left&space;(&space;x,&space;W,&space;V,&space;b,&space;c&space;\right&space;)=&space;Swish_b(xW&space;&plus;&space;b)&space;\oplus&space;(xV&space;&plus;&space;b)">

* [Source Paper : GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)

8. SeGLU:

    SeGLU is an activation function which is a variant of GLU. 

9. ReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\0&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ReLU.png"> 
</p>

* [Source Paper : Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines." In Icml. 2010.](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

10. Identity:

    $f(x) = x$

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Identity.png"> 
</p>

11. Step:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}1&space;&&space;x&space;<&space;0\\0&space;&&space;x&space;\geq&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Step.png"> 
</p>

12. Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Sigmoid.png"> 
</p>

* [Source Paper : Han, Jun, and Claudio Moraga. "The influence of the sigmoid function parameters on the speed of backpropagation learning." In International workshop on artificial neural networks, pp. 195-201. Springer, Berlin, Heidelberg, 1995.](https://link.springer.com/chapter/10.1007/3-540-59497-3_175)

13. Hard Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,&space;min(1,\frac{x&plus;1}{2}))">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/HardSigmoid.png"> 
</p>

* [Source Paper : Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Binaryconnect: Training deep neural networks with binary weights during propagations." Advances in neural information processing systems 28 (2015).](https://arxiv.org/abs/1511.00363)

14. Log Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?LogSigmoid(x)=\log\left(\dfrac{1}{1&plus;\exp(-x_i)}\right)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/LogSigmoid.png"> 
</p>

15. SiLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x(\frac{1}{1&plus;e^{-x}})">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SiLU.png"> 
</p>

* [Source Paper : Elfwing, Stefan, Eiji Uchibe, and Kenji Doya. "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning." Neural Networks 107 (2018): 3-11.](https://arxiv.org/abs/1702.03118)

16. ParametricLinear:

    $f(x) = a*x$

17. PiecewiseLinear:

    Choose some xmin and xmax, which is our "range". Everything less than than this range will be 0, and everything greater than this range will be 1. Anything else is linearly-interpolated between.

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\begin{cases}0&space;&&space;x&space;<&space;x_{min}\\&space;mx&space;&plus;&space;b&space;&&space;x_{min}&space;<&space;x&space;<&space;x_{max}\\&space;1&space;&&space;x&space;>&space;x_{xmax}&space;\end{cases}">
    
    
<img src="https://latex.codecogs.com/svg.image?m&space;=&space;\frac{1}{x_{max}&space;-&space;x_{min}}">
    
    
<img src="https://latex.codecogs.com/svg.image?b&space;=&space;-mx_{min}&space;=&space;1&space;-&space;mx_{max}">

<p align="center"> 
  <img width="700" height="400" src="https://i.stack.imgur.com/cguIH.png"> 
</p>

18. Complementary Log-Log (CLL):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;1-e^{-e^x}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Complementary%20Log-Log.png"> 
</p>

* [Source Paper : Gomes, Gecynalda S. da S., and Teresa B. Ludermir. "Complementary log-log and probit: activation functions implemented in artificial neural networks." In 2008 Eighth International Conference on Hybrid Intelligent Systems, pp. 939-942. IEEE, 2008.](https://www.computer.org/csdl/proceedings-article/his/2008/3326a939/12OmNxHrykP)

19. Bipolar:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}-1&space;&&space;x&space;\leq&space;0\\1&space;&&space;x&space;>&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Bipolar.png"> 
</p>

20. Bipolar Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{1-e^{-x}}{1&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/BipolarSigmoid.png"> 
</p>

* [Source Paper : Mansor, Mohd Asyraf, and Saratha Sathasivam. "Activation function comparison in neural-symbolic integration." In AIP Conference Proceedings, vol. 1750, no. 1, p. 020013. AIP Publishing LLC, 2016.](https://aip.scitation.org/doi/10.1063/1.4954526)

21. Tanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{e^{x}-e^{-x}}{e^{x}&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/tanh.png"> 
</p>

* [Source Paper : Harrington, Peter de B. "Sigmoid transfer functions in backpropagation neural networks." Analytical Chemistry 65, no. 15 (1993): 2167-2168.](https://pubs.acs.org/doi/pdf/10.1021/ac00063a042)

22. Tanh Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x-tanh(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/tanhShrink.png"> 
</p>

23. LeCunTanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;1.7159\&space;tanh(\frac{2}{3}x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/LeCunTanh.png"> 
</p>

* [Source Paper : LeCun, Yann A., Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller. "Efficient backprop." In Neural networks: Tricks of the trade, pp. 9-48. Springer, Berlin, Heidelberg, 2012.](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

24. Hard Tanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}-1&space;&&space;x&space;<&space;-1\\x&space;&&space;-1&space;\leq&space;x&space;\leq&space;1\\1&space;&&space;x&space;>&space;1&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/HardTanh.png"> 
</p>

25. TanhExp:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(e^x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/TanhExp.png"> 
</p>

* [Source Paper : Liu, Xinyu, and Xiaoguang Di. "TanhExp: A smooth activation function with high convergence speed for lightweight neural networks." IET Computer Vision 15, no. 2 (2021): 136-150.](https://arxiv.org/abs/2003.09855)

26. ABS:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;|x|">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Abs.png"> 
</p>

27. SquaredReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x^2&space;&&space;x&space;\geq&space;0\\0&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SquaredReLU.png"> 
</p>

* [Source Paper : So, David, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V. Le. "Searching for Efficient Transformers for Language Modeling." Advances in Neural Information Processing Systems 34 (2021): 6010-6022.](https://arxiv.org/abs/2109.08668)

28. ParametricReLU (PReLU):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;\alpha&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/PReLU.png">
</p>

* [Source Paper : He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." In Proceedings of the IEEE international conference on computer vision, pp. 1026-1034. 2015.](https://arxiv.org/abs/1502.01852)

29. RandomizedReLU (RReLU):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;\alpha&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

    <img src="https://latex.codecogs.com/svg.image?a&space;\sim&space;U(l,u)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/RReLU.png">   
</p>

* [Source Paper : Xu, Bing, Naiyan Wang, Tianqi Chen, and Mu Li. "Empirical evaluation of rectified activations in convolutional network." arXiv preprint arXiv:1505.00853 (2015).](https://arxiv.org/abs/1505.00853?context=cs)

30. LeakyReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;0.01&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/LeakyReLU.png">
</p>

31. ReLU6:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;min(6,&space;max(0,x))">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ReLU6.png">  
</p>

* [Source Paper : Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).](https://arxiv.org/abs/1704.04861)

32. ModReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}(|x|&plus;b)\frac{x}{|x|}&space;&&space;|x|&plus;b&space;\geq&space;0&space;\\0&space;&&space;|x|&plus;b&space;\leq&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ModReLU.png">
</p>

* [Source Paper : Arjovsky, Martin, Amar Shah, and Yoshua Bengio. "Unitary evolution recurrent neural networks." In International conference on machine learning, pp. 1120-1128. PMLR, 2016.](https://arxiv.org/abs/1511.06464?context=stat)

33. CosReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,x)&space;&plus;&space;cos(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/CosRelu.png">    
</p>

34. SinReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,x)&space;&plus;&space;sin(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SinReLU.png">    
</p>

35. Probit:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\sqrt{2}\&space;\&space;erfinv(2x&space;-&space;1)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Probit.png">    
</p>

36. Cosine:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;Cos(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Cosine.png">    
</p>

37. Gaussian:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;e^{-\frac{1}{2}x^2}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Gaussian.png">    
</p>

38. Multiquadratic:

    Choose some point (x,y).
    
    <img src="https://latex.codecogs.com/svg.image?\rho(z)&space;=&space;\sqrt{(z&space;-&space;x)^{2}&space;&plus;&space;y^{2}}&space;">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Multiquadratic.png">    
</p>

39. InvMultiquadratic:

    <img src="https://latex.codecogs.com/svg.image?\rho(z)&space;=&space;\frac{1}{\sqrt{(z&space;-&space;x)^{2}&space;&plus;&space;y^{2}}&space;}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/InvMultiquadratic.png">    
</p>

40. SoftPlus:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;ln(1&plus;e^x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SoftPlus.png">    
</p>

* [Source Paper : Dugas, Charles, Yoshua Bengio, François Bélisle, Claude Nadeau, and René Garcia. "Incorporating second-order functional knowledge for better option pricing." Advances in neural information processing systems 13 (2000).](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.966.2210&rep=rep1&type=pdf)

41. Mish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(SoftPlus(x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Mish.png">    
</p>

* [Source Paper : Misra, Diganta. "Mish: A self regularized non-monotonic neural activation function." arXiv preprint arXiv:1908.08681 4, no. 2 (2019): 10-48550.](https://arxiv.org/abs/1908.08681)

42. Smish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(log(1&plus;Sigmoid(x)))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Smish.png">    
</p>

43. ParametricSmish (PSmish):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;a\&space;tanh(log(1&plus;Sigmoid(b)))">
    
    <img src="https://latex.codecogs.com/svg.image?a=&space;\alpha&space;x">
    
    <img src="https://latex.codecogs.com/svg.image?b=&space;\beta&space;x">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/PSmish.png">    
</p>

44. Swish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-e^{-\beta&space;x}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Swish.png">    
</p>

* [Source Paper : Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions." arXiv preprint arXiv:1710.05941 (2017).](https://arxiv.org/abs/1710.05941?context=cs.LG)

45. ESwish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\beta\&space;\frac{x}{1-e^{-\beta&space;x}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ESwish.png">    
</p>

* [Source Paper : Alcaide, Eric. "E-swish: Adjusting activations to different network depths." arXiv preprint arXiv:1801.07145 (2018).](https://arxiv.org/pdf/1801.07145)

46. Hard Swish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;\frac{ReLU6(x&plus;3)}{6}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/HardSwish.png">    
</p>

* [Source Paper : Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1314-1324. 2019.](https://ieeexplore.ieee.org/document/9008835)

47. GCU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;cos(x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/GCU.png">    
</p>

* [Source Paper : Noel, Mathew Mithra, Advait Trivedi, and Praneet Dutta. "Growing cosine unit: A novel oscillatory activation function that can speedup training and reduce parameters in convolutional neural networks." arXiv preprint arXiv:2108.12943 (2021).](https://deepai.org/publication/growing-cosine-unit-a-novel-oscillatory-activation-function-that-can-speedup-training-and-reduce-parameters-in-convolutional-neural-networks)

48. CoLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-x&space;e^{-(x&plus;e^x)}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/CoLU.png">    
</p>

* [Source Paper : Vagerwal, Advait. "Deeper Learning with CoLU Activation." arXiv preprint arXiv:2112.12078 (2021).](https://arxiv.org/abs/2112.12078)

49. PELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}cx&space;&&space;x&space;>&space;0\\&space;\alpha&space;e^{\frac{x}{b}}-1&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/PELU.png">    
</p>

* [Source Paper : Trottier, Ludovic, Philippe Giguere, and Brahim Chaib-Draa. "Parametric exponential linear unit for deep convolutional neural networks." In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA), pp. 207-214. IEEE, 2017.](https://arxiv.org/abs/1605.09332?context=cs)

50. SELU:

    <img src="https://latex.codecogs.com/svg.image?f\left(x\right)&space;=&space;\lambda{x}&space;\text{&space;if&space;}&space;x&space;\geq{0}$$&space;$$f\left(x\right)&space;=&space;\lambda{\alpha\left(\exp\left(x\right)&space;-1&space;\right)}&space;\text{&space;if&space;}&space;x&space;<&space;0">
    
    where $\alpha \approx 1.6733$ & $\lambda \approx 1.0507$
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/SELU.png">    
</p>

* [Source Paper : Klambauer, Günter, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. "Self-normalizing neural networks." Advances in neural information processing systems 30 (2017).](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)

51. CELU:

    <img src="https://latex.codecogs.com/svg.image?CELU\left&space;(&space;x&space;\right&space;)=&space;max(0,&space;x)&space;&plus;&space;min(0&space;,&space;\alpha&space;(e^{\frac{x}{\alpha&space;}}&space;-1))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/CELU.png">    
</p>

* [Source Paper : Barron, Jonathan T. "Continuously differentiable exponential linear units." arXiv preprint arXiv:1704.07483 (2017).](https://arxiv.org/abs/1704.07483)

52. ArcTan:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;ArcTang(x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ArcTan.png">    
</p>

53. ShiftedSoftPlus:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;log(0.5&plus;0.5e^x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ShiftedSoftPlus.png">    
</p>

* [Source Paper : Schütt, Kristof, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in neural information processing systems 30 (2017).](https://dl.acm.org/doi/abs/10.5555/3294771.3294866)

54. Softmax:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x_i}{\sum_j&space;x_j}">

* [Source Paper : Gold, Steven, and Anand Rangarajan. "Softmax to softassign: Neural network algorithms for combinatorial optimization." Journal of Artificial Neural Networks 2, no. 4 (1996): 381-399.](https://www.cise.ufl.edu/~anand/pdf/jannsub.pdf)

55. Logit:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-x}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Logit.png">    
</p>

56. GELU:

    <img src="https://latex.codecogs.com/svg.image?f(X)&space;=&space;x&space;\&space;\phi&space;\(&space;x&space;)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/GELU.png">    
</p>

57. Softsign:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{|x|&plus;1}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Softsign.png">    
</p>

58. ELiSH:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}\frac{x}{1&plus;e^{-x}}&space;&&space;x&space;\geq&space;0\\&space;\frac{e^x-1}{1&plus;e^{-x}}&space;&&space;x&space;<&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ELiSH.png">    
</p>

* [Source Paper : Basirat, Mina, and Peter M. Roth. "The quest for the golden activation function." arXiv preprint arXiv:1808.00783 (2018).](https://arxiv.org/abs/1808.00783)

59. Hard ELiSH:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x\&space;max(0,min(1,\frac{x&plus;1}{2}))&space;&&space;x&space;\geq&space;0\\&space;(e^x-1)&space;max(0,min(1,\frac{x&plus;1}{2}))&space;&&space;x&space;<&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/HardELiSH.png">    
</p>

* [Source Paper : Basirat, Mina, and Peter M. Roth. "The quest for the golden activation function." arXiv preprint arXiv:1808.00783 (2018).](https://arxiv.org/abs/1808.00783)

60. Serf:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;erf(ln(1&plus;e^x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Serf.png">    
</p>

* [Source Paper : Nag, Sayan, and Mayukh Bhattacharyya. "SERF: Towards better training of deep neural networks using log-Softplus ERror activation Function." arXiv preprint arXiv:2108.09598 (2021).](https://arxiv.org/abs/2108.09598?context=cs)

61. ELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;\alpha&space;(exp(x)-1)&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/ELU.png">    
</p>

* [Source Paper : Clevert, Djork-Arné, Thomas Unterthiner, and Sepp Hochreiter. "Fast and accurate deep network learning by exponential linear units (elus)." arXiv preprint arXiv:1511.07289 (2015).](https://dblp.org/rec/journals/corr/ClevertUH15)

62. Phish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(gelu(x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/Phish.png">    
</p>

* [Source Paper : Naveen, Philip. "Phish: A novel hyper-optimizable activation function." (2022).](https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824/2)

63. QReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;0.01\&space;x(x-2)&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/QReLU.png">    
</p>

* [Source Paper : Parisi, Luca, Daniel Neagu, Renfei Ma, and Felician Campean. "QReLU and m-QReLU: Two novel quantum activation functions to aid medical diagnostics." arXiv preprint arXiv:2010.08031 (2020).](https://arxiv.org/abs/2010.08031)

64. m-QReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;0.01\&space;x&space;-x&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/m-QReLU.png">    
</p>

* [Source Paper : Parisi, Luca, Daniel Neagu, Renfei Ma, and Felician Campean. "QReLU and m-QReLU: Two novel quantum activation functions to aid medical diagnostics." arXiv preprint arXiv:2010.08031 (2020).](https://arxiv.org/abs/2010.08031)

65. FReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&plus;b&space;&&space;x&space;>&space;0\\&space;b&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/raw/master/images/FReLU.png">    
</p>

* [Source Paper : Qiu, Suo, Xiangmin Xu, and Bolun Cai. "FReLU: flexible rectified linear units for improving convolutional neural networks." In 2018 24th international conference on pattern recognition (icpr), pp. 1223-1228. IEEE, 2018.](https://arxiv.org/abs/1706.08098)

## Cite this repository

```sh
@software{Pouya_ActTensor_2022,
author = {Pouya, Ardehkhani and Pegah, Ardehkhani},
license = {MIT},
month = {7},
title = {{ActTensor}},
url = {https://github.com/pouyaardehkhani/ActTensor},
version = {1.0.0},
year = {2022}
}
```
