<div align="center">
  <img src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ActTensor%20logo.png"><br>
</div>

# **ActTensor**

## **What is it?**

ActTensor is a Python package that provides state-of-the-art activation functions which facilitate using them in Deep Learning projects in an easy and fast manner. 

## **Why not using tf.keras.activations?**
As you may know, TensorFlow only has a few defined activation functions and most importantly it does not include newly-introduced activation functions. Wrting another one requires time and energy; however, This package has most of the widely-used, and even state-of-the-art activation functions that are ready to use in your models.


## **Which activation it supports?**

1. Soft Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x-\lambda&space;&&space;x&space;>&space;\lambda\\&space;x&plus;\lambda&space;&&space;x&space;<&space;-\lambda\\&space;0&space;&&space;otherwise&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SoftShrink.png"> 
</p>

2. Hard Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\begin{cases}x&space;&&space;x&space;>&space;\lambda\\&space;x&space;&&space;x&space;<&space;-\lambda\\&space;0&space;&&space;otherwise&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardShrink.png"> 
</p>

3. GLU:

4. Bilinear:

5. ReGLU:

6. GeGLU:

7. SwiGLU:

8. SeGLU:

9. ReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\0&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ReLU.png"> 
</p>

10. Identity:

    $f(x) = x$

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Identity.png"> 
</p>

11. Step:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}1&space;&&space;x&space;<&space;0\\0&space;&&space;x&space;\geq&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Step.png"> 
</p>

12. Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Sigmoid.png"> 
</p>

13. Hard Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,&space;min(1,\frac{x&plus;1}{2}))">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardSigmoid.png"> 
</p>

14. Log Sigmoid:

    <img src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/LogSigmoid.png">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardSigmoid.png"> 
</p>

15. SiLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x(\frac{1}{1&plus;e^{-x}})">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SiLU.png"> 
</p>

16. ParametricLinear:

17. PiecewiseLinear:

18. Complementary Log-Log (CLL):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;1-e^{-e^x}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Complementary%20Log-Log.png"> 
</p>

19. Bipolar:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}-1&space;&&space;x&space;\leq&space;0\\1&space;&&space;x&space;>&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Bipolar.png"> 
</p>

20. Bipolar Sigmoid:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{1-e^{-x}}{1&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/BipolarSigmoid.png"> 
</p>

21. Tanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{e^{x}-e^{-x}}{e^{x}&plus;e^{-x}}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/tanh.png"> 
</p>

22. Tanh Shrink:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x-tanh(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/tanhShrink.png"> 
</p>

23. LeCunTanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;1.7159\&space;tanh(\frac{2}{3}x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/LeCunTanh.png"> 
</p>

24. Hard Tanh:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}-1&space;&&space;x&space;<&space;-1\\x&space;&&space;-1&space;\leq&space;x&space;\leq&space;1\\1&space;&&space;x&space;>&space;1&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardTanh.png"> 
</p>

25. TanhExp:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(e^x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/TanhExp.png"> 
</p>

26. ABS:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;|x|">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Abs.png"> 
</p>

27. SquaredReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x^2&space;&&space;x&space;\geq&space;0\\0&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SquaredReLU.png"> 
</p>

28. ParametricReLU (PReLU):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;\alpha&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/PReLU.png">
</p>

29. RandomizedReLU (RReLU):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;\alpha&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

    <img src="https://latex.codecogs.com/svg.image?a&space;\sim&space;U(l,u)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/RReLU.png">   
</p>

30. LeakyReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;\geq&space;0\\&space;0.01&space;x&space;&&space;x&space;<&space;0&space;\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/LeakyReLU.png">
</p>

31. ReLU6:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;min(6,&space;max(0,x))">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ReLU6.png">  
</p>

32. ModReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}(|x|&plus;b)\frac{x}{|x|}&space;&&space;|x|&plus;b&space;\geq&space;0&space;\\0&space;&&space;|x|&plus;b&space;\leq&space;0\end{cases}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ModReLU.png">
</p>

33. CosReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,x)&space;&plus;&space;cos(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/CosRelu.png">    
</p>

34. SinReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,x)&space;&plus;&space;sin(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SinReLU.png">    
</p>

35. Probit:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\sqrt{2}\&space;\&space;erfinv(2x&space;-&space;1)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Probit.png">    
</p>

36. Cosine:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;Cos(x)">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Cosine.png">    
</p>

37. Gaussian:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;e^{-\frac{1}{2}x^2}">

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Gaussian.png">    
</p>

38. Multiquadratic:

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Multiquadratic.png">    
</p>

39. InvMultiquadratic:

<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/InvMultiquadratic.png">    
</p>

40. SoftPlus:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;ln(1&plus;e^x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SoftPlus.png">    
</p>

41. Mish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(SoftPlus(x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Mish.png">    
</p>

42. Smish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(log(1&plus;Sigmoid(x)))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Smish.png">    
</p>

43. ParametricSmish (PSmish):

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;a\&space;tanh(log(1&plus;Sigmoid(b)))">
    
    <img src="https://latex.codecogs.com/svg.image?a=&space;\alpha&space;x">
    
    <img src="https://latex.codecogs.com/svg.image?b=&space;\beta&space;x">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/PSmish.png">    
</p>

44. Swish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-e^{-\beta&space;x}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Swish.png">    
</p>

45. ESwish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\beta\&space;\frac{x}{1-e^{-\beta&space;x}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ESwish.png">    
</p>

46. Hard Swish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;\frac{ReLU6(x&plus;3)}{6}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardSwish.png">    
</p>

47. GCU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;cos(x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/GCU.png">    
</p>

48. CoLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-x&space;e^{-(x&plus;e^x)}}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/CoLU.png">    
</p>

49. PELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}cx&space;&&space;x&space;>&space;0\\&space;\alpha&space;e^{\frac{x}{b}}-1&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/PELU.png">    
</p>

50. SELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;scale\&space;\times&space;(max(0,x)&plus;min(0,\alpha∗(e^x−1)))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/SELU.png">    
</p>

51. CELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;max(0,x)&plus;min(0,&space;\alpha&space;(exp(\frac{x}{\alpha})−1))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/CELU.png">    
</p>

52. ArcTan:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;ArcTang(x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ArcTan.png">    
</p>

53. ShiftedSoftPlus:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;log(0.5&plus;0.5e^x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ShiftedSoftPlus.png">    
</p>

54. Softmax:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x_i}{\sum_j&space;x_j}">
    
55. Logit:

    <img src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Logit.png">
    
<p align="center"> 
  <img width="700" height="400" src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{1-x}">    
</p>

56. GELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;Φ(x)">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/GELU.png">    
</p>

57. Softsign:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;\frac{x}{|x|&plus;1}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Softsign.png">    
</p>

58. ELiSH:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}\frac{x}{1&plus;e^{-x}}&space;&&space;x&space;\geq&space;0\\&space;\frac{e^x-1}{1&plus;e^{-x}}&space;&&space;x&space;<&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ELiSH.png">    
</p>

59. Hard ELiSH:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x\&space;max(0,min(1,\frac{x&plus;1}{2}))&space;&&space;x&space;\geq&space;0\\&space;(e^x-1)&space;max(0,min(1,\frac{x&plus;1}{2}))&space;&&space;x&space;<&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/HardELiSH.png">    
</p>

60. Serf:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;erf(ln(1&plus;e^x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Serf.png">    
</p>

61. ELU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;\alpha&space;(exp(x)-1)&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/ELU.png">    
</p>

62. Phish:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=&space;x\&space;tanh(gelu(x))">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/Phish.png">    
</p>

63. QReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;0.01\&space;x(x-2)&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/QReLU.png">    
</p>

64. m-QReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&space;&&space;x&space;>&space;0\\&space;0.01\&space;x&space;-x&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/m-QReLU.png">    
</p>

65. FReLU:

    <img src="https://latex.codecogs.com/svg.image?f(x)&space;=\begin{cases}x&plus;b&space;&&space;x&space;>&space;0\\&space;b&space;&&space;x&space;\leq&space;0&space;\end{cases}">
    
<p align="center"> 
  <img width="700" height="400" src="https://github.com/pouyaardehkhani/ActTensor/blob/master/images/FReLU.png">    
</p>
