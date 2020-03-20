# Homework 2 Write-ups

### Chenxi Xiang

---

1. What is the form of sigmoid function $σ(z)$ ? Show that $σ′(z) = σ(z)[1 − σ(z)]$.

$$
\sigma(z)=\frac{1}{1 + e^{-x}} \\
  \sigma'(x) = -\frac{1}{(1+e^{-x})^2} \times(-e^{-x}) \\
  = \frac{1}{1+e^{-x}} \times \frac{e^{-x}}{1+e^{-x}} \\ 
  = \sigma(z)[1-\sigma(z)]
$$

2. Another popular activation function is $tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}$ , show that $tanh′(z) = 1 − tanh(z)^2$.

$$
tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}=1-\frac{2}{e^{2x}+1} \\tanh'(z)=\frac{4e^{2x}}{(e^{2x}+1)^2}=\frac{4}{e^{2x}+1}-\frac{4}{(e^{2x}+1)^2}\\=1 − tanh(z)^2
$$

3. For a single variable single layer perceptron with sigmoid activation function (equivalent

to LR) and loss function defined as:

$$
\hat{y}_i = σ(w_1x_i+w_0)\\L(w_0, w_1) = \sum_i y_i lg(\hat{y}_i)+(1−y_i)lg(1−\hat{y}_i)
$$

​		Show that:

$$
\frac{∂L}{∂w_1} =\sum_i(y_i−\hat{y}_i)x_i\\\frac{∂L}{∂w_0} =\sum_i(y−\hat{y}_i)
$$


By Q1's conclusion, we have : $σ′(w_1x_i+w_0) = σ(w_1x_i+w_0)[1 − σ(w_1x_i+w_0)]x_i$
$$
\frac{∂L}{∂w_1} =\sum_iy_i\frac{1}{\sigma(w_1x_i+w_0)}\sigma(w_1x_i+w_0)(1-\sigma(w_1x_i+w_0))x_i-\\(1-y_i)\frac{1}{\sigma(w_1x_i+w_0)}\sigma(w_1x_i+w_0)(1-\sigma(w_1x_i+w_0))x_i\\=\sum_i y_i(1-\sigma(w_1x_i+w_0))x_i - (1-y_i)\sigma(w_1x_i+w_0)x_i\\=\sum_ix_iy_i-\sigma(w_1x_i+w_0)x_i=\sum_ix_i(y_i-\hat{y_i})
$$

Similarly:
$$
\frac{∂L}{∂w_0} =\sum_iy_i\frac{1}{\sigma(w_1x_i+w_0)}\sigma(w_1x_i+w_0)(1-\sigma(w_1x_i+w_0))-\\(1-y_i)\frac{1}{\sigma(w_1x_i+w_0)}\sigma(w_1x_i+w_0)(1-\sigma(w_1x_i+w_0))\\=\sum_i y_i(1-\sigma(w_1x_i+w_0)) - (1-y_i)\sigma(w_1x_i+w_0)\\=\sum_iy_i-\sigma(w_1x_i+w_0)=\sum_i(y_i-\hat{y_i})
$$

4. For column vectors $\vec{x}$ and $\vec{w}$ , and a symmetric matrix $\overleftrightarrow{M}$, define the gradient operator:

$∇_\vec{x} = (\frac{∂}{∂x_0}, \frac{∂}{∂x_1}, ...,\frac{∂}{∂x_n})^T$

show that:

$∇_x(\vec{w}^T\vec{x}) = \vec{w}$
$∇_x(\vec{x}^T\vec{w}) = \vec{w}$
$∇_x(\vec{w}^T\overleftrightarrow{M}\vec{x}) = \overleftrightarrow{M}\vec{w}$



We have $w^Tx=x^Tw$, so the first and second conclusions are same. And for each pair of $x_i$ and $w_i$, we have $\frac{∂}{∂x}=w$. When we take the gradient of linear product, it is same to take gradient for every pair and concate them. So $∇_x(\vec{w}^T\vec{x}) = \vec{w}$ and $∇_x(\vec{x}^T\vec{w}) = \vec{w}$.

For the third, we can take the elemental-wise gradient too, and just take the quadratic way. 



5. Let’s expand Q3 to a more general case. Suppose there is a single layer perceptron with multiple variables:

$\hat{y}_i = σ( \vec{w}^T \vec{x_i} )$
$L(\vec{w}) = \sum_i y_i lg(\hat{y}_i)+(1−y_i)lg(1−\hat{y}_i)$



show that:

$∇_\vec{w}L(\vec{w}) = \sum_i(y_i - \hat{y}_i)\vec{x_i}$

As it is linear for $x_i$, for each pair of $x_i, x_j$, we can think that $x_i$is a variable and other x are constant. so for each $x_i$, $∇_\vec{w}L(\vec{w})=\sum_i(y_i-\hat{y}_i)x_i$. So we have the conclusion:
$$
∇_\vec{w}L(\vec{w}) = \sum_i(y_i - \hat{y}_i)\vec{x_i}
$$

6. In a CNN illustrated as Fig 1, suppose the loss function is:

$L(\overleftrightarrow{U}, \vec{w}) = \sum_i y_i lg(\hat{y}_i)+(1−y_i)lg(1−\hat{y}_i)$

From the conclusion in Q5, we can get that:

$∇_w L(\overleftrightarrow{U}, \vec{w}) = \sum_i (y_i -\hat{y}_i)\vec{h}^{(i)}$



Can you calculate $∇_{u_i} L(U,w)$ using similar techniques?

<img src="E:/0.OneDrive/OneDrive/1.Code/1.Github/NLP-HW-2020/HW02-CNN/CNN.png" style="width:700px">

For each i, we have $w = [w_0,w_1,\cdots,w_f,\cdots,w_F]^T$, thus we have:
$$
(y_ilog(\sigma))^\prime_f=y_i\frac{1}{\sigma}\sigma(1-\sigma)(w^Th)^{\prime}_f=y_i(1-\sigma)w_f(h_f)^\prime_f\\=y_i(1-\sigma)w_f(max[tanh(u_f^Tx_1),\cdots,tanh(u_f^Tx_{n-k+1})])^\prime_f \\=y_i(1-\sigma)w_f(tanh_{max})^\prime =y_i(1-\sigma)w_f(1-tanh_{max}^2)\vec{x}_{argmax}
$$
So we can have:
$$
∇_{u_f} L(U,w) = \sum_i(y_i-\sigma)w_f(1-tanh_{max}^2)\vec{x}_{argmax}
$$

---

Answers about coding:

The CNN performs no better predictive power than the bag of words model. It seems that a more carefully designed network is needed for this kind of work. The number of layers, the activation functions and the structure of entire network should be improved regarding to different kinds of problems.