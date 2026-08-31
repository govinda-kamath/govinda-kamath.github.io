---
layout: post
title: Gaussian Processes, Part 1 — Regression
date: 2026-08-30
description: "From Bayesian linear regression to Gaussian processes: the weight-space view, the kernel trick, and the function-space view."
slug: gp-1-regression
series: gaussian-processes
series_part: 1
tags: gaussian-processes machine-learning
categories: gaussian-processes
---

I have been reading the classical book on Gaussian Processes by Rasmussen and Williams
[[pdf](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)] over the last few weekends. I
had meant to read this for a long time and only could get to it now. I am thus jotting down my notes
here as I started this blog recently. The main purpose of this is to help me digest the material, but
hopefully it will be useful to others as well.

A Gaussian Process is usually thought of as defining a distribution over functions, where we 
can think of inference taking place directly over the space of functions. However to start off
we start with the more basic view of Gaussians -- the _weight space_ view of Gaussians (as opposed to the 
_function space_ view) to build some intuition.

## Weight-Space View

We start with a simple linear regression model

### The Standard Linear Model

**The setup**: Let us assume we have a training set $\mathcal{D}$ of $n$ variables,

$$
\begin{align}
\mathcal{D} = \{ (\mathbf{x}_i, y_i) | i=1,\cdots,n \},
\end{align}
$$

where $\mathbf{x}_i \in \mathbb{R}^D$ are $D$ dimensional covariates
and $y_i \in \mathbb{R}$ is a scalar output or target variable. More compactly,
we can represent 

$$
\begin{align*}
X &= [\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_n] \in \mathbb{R}^{D \times n},\\
y &= (y_1, y_2, \cdots, y_n)^{\top} \in \mathbb{R}^n,
\end{align*}
$$

and thus represent the data as 

$$
\begin{align*}
\mathcal{D} &= (X,y)
\end{align*}
$$

The standard linear model then can be stated as 

$$
\begin{align*}
f(\mathbf{x}_i) &= \mathbf{x}_i^{\top}\mathbf{w}, \text{ for some } \mathbf{w} \in \mathbb{R}^D,\\
y_i &= f(\mathbf{x}_i) + \varepsilon, \text{ where}, \\
\varepsilon &\sim \mathcal{N}(0, \sigma_n^2), \text{ for some variance }\sigma_n^2
\end{align*}
$$

More compactly, overloading the definition of $f$ to the vector case, we have that

$$
\begin{align}
\mathbf{y} &= f(X) + \varepsilon = X^\top\mathbf{w} + \varepsilon,
\end{align}
$$

where $\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$.

This definition now directly gives us that

$$
\begin{align}
\mathbf{y} \big| X, \mathbf{w} &\sim \mathcal{N}(X^\top \mathbf{w}, \sigma_n^2 I), \label{eq:raw-y-likelihood}\\
\implies p\left(\mathbf{y} \big| X, \mathbf{w}\right) &= 
\frac{1}{(2 \pi \sigma_n^2)^{\frac{n}{2}}}  \exp\left( - \frac{1}{2 \sigma_n^2} \|\mathbf{y} -X^\top \mathbf{w} \|^2 \right ) \nonumber
\end{align} 
$$

To start off our exploration, we place a prior on $\mathbf{w}$. The natural choice is 

$$
\begin{align}
\mathbf{w} &\sim \mathcal{N}(0, \Sigma_p), \label{eq:w-prior}
\end{align}
$$

for some $\Sigma_p \in \mathbb{R}^{D \times D}$.

The first order of business is to determine the _maximum a posteriori_ (MAP)
estimate of $\mathbf{w} \mid X, \mathbf{y}$. To do this we start by noting that, as 
$(\mathbf{y}, \mathbf{w}) \mid X$ is jointly Gaussian, and thus $\mathbf{w} \mid X, \mathbf{y}$ is
Gaussian because conditional distributions of jointly Gaussian random variables are also 
Gaussian [[wiki](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions)].
Then we proceed by noting that,

$$
\begin{align*}
p(\mathbf{w} | X, \mathbf{y}) &= \frac{p(\mathbf{w},\mathbf{y} | X)}{p (\mathbf{y} | X)}, \\
&= \frac{p(\mathbf{y} | \mathbf{w}, X) p (\mathbf{w}| X)}{p (\mathbf{y} | X)},\\
&\overset{(a)}{=}  \frac{p(\mathbf{y} | \mathbf{w}) p (\mathbf{w})}{p (\mathbf{y} | X)},\\
&\overset{(b)}{\propto} p(\mathbf{y} | \mathbf{w}) p (\mathbf{w})
\end{align*}
$$

where $(a)$ follows from the fact that conditioned on $\mathbf{w}$, $\mathbf{y}$ is independent of $X$, and that
$\mathbf{w}$ is independent of $X$. $(b)$ follows from dropping the denominator as it is independent of $\mathbf{w}$.

#### Completing the square to determine Gaussian distributions

One trick we will return to again and again is completing the square to determine Gaussian distributions. The idea: if we
know a random variable is Gaussian, we only need to match the quadratic and linear terms in the exponent of its density —
the quadratic term gives the precision matrix and the linear term gives the mean.

That is, if we have that $\mathbf{w} \sim \mathcal{N} (\bar{\mathbf{w}}, A_{w})$. 

$$
\begin{align}
p(\mathbf{w}) &\propto \exp\left( - \frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^\top A_{w}^{-1} (\mathbf{w}-\bar{\mathbf{w}})\right), \nonumber \\
&= \exp\left(-\frac{1}{2} \mathbf{w}^\top A_{w}^{-1} \mathbf{w} + \mathbf{w}^\top A_{w}^{-1} \bar{\mathbf{w}} - \frac{1}{2} \bar{\mathbf{w}}^\top A_{w}^{-1} \bar{\mathbf{w}}  \right), \nonumber \\
&\propto \exp\left(-\frac{1}{2} \mathbf{w}^\top A_{w}^{-1} \mathbf{w} + \mathbf{w}^\top A_{w}^{-1} \bar{\mathbf{w}} \right) \label{eq:complete-square}
\end{align}
$$

If we can get the quadratic and linear terms in $\mathbf{w}$ of the exponential, we can then match terms to get the 
whole Gaussian distribution.

Let us see this trick in action to determine $p(\mathbf{w} \mid X, \mathbf{y})$. We have that,

$$
\begin{align}
p(\mathbf{w} | X, \mathbf{y}) &\propto p(\mathbf{y} | \mathbf{w}) p (\mathbf{w}), \\
&\overset{(a)}{\propto} \exp\left( - \frac{1}{2 \sigma_n^2} \|\mathbf{y} -X^\top \mathbf{w} \|^2 \right ) \exp \left( -\frac{1}{2} \mathbf{w}^\top \Sigma_p^{-1} \mathbf{w}\right), \nonumber\\
&= \exp \left( -\frac{1}{2} \mathbf{w}^\top \left[ \frac{1}{\sigma_n^2} XX^\top  + \Sigma_p^{-1} \right] \mathbf{w}
+ \frac{1}{\sigma_n^2} \mathbf{w}^\top X \mathbf{y} - \frac{1}{\sigma_n^2} \mathbf{y}^\top \mathbf{y}
\right),\nonumber\\
&\propto \exp \left( -\frac{1}{2} \mathbf{w}^\top \left[ \frac{1}{\sigma_n^2} XX^\top  + \Sigma_p^{-1} \right] \mathbf{w}
+ \frac{1}{\sigma_n^2} \mathbf{w}^\top X \mathbf{y}
\right) \label{eq:map-complete-square}
\end{align}
$$

where $(a)$ follows from \eqref{eq:raw-y-likelihood} and \eqref{eq:w-prior}.
Now comparing quadratic terms of eq. \eqref{eq:complete-square} and eq. \eqref{eq:map-complete-square}, 
we have that

$$
\begin{align}
A_w^{-1} &= \left[ \frac{1}{\sigma_n^2} XX^\top  + \Sigma_p^{-1} \right] := A, \\
\end{align}
$$

giving us the covariance matrix. Comparing the linear terms we have that

$$
\begin{align}
A \bar{\mathbf{w}} &= \frac{1}{\sigma_n^2} X \mathbf{y}, \nonumber\\
\implies \bar{\mathbf{w}} &= \frac{1}{\sigma_n^2} A^{-1} X \mathbf{y}.
\end{align}
$$

This gives us that 

$$
\begin{align}
\mathbf{w} \mid X, \mathbf{y} &\sim \mathcal{N}\left(\frac{1}{\sigma_n^2} A^{-1} X \mathbf{y}, A^{-1}\right),\label{eq:map-w}
\end{align}
$$

with $A = \frac{1}{\sigma_n^2} XX^\top  + \Sigma_p^{-1}$ giving us the MAP 
estimate of $\mathbf{w}$ conditioned on the data $\mathcal{D}$. 

One point to note is that if $\Sigma_p$ were the identity matrix,
then the MAP estimate would be 
$$\frac{1}{\sigma_n^2} \left[\frac{1}{\sigma_n^2} XX^\top  + I \right]^{-1} X \mathbf{y} = \left[ XX^\top  + \sigma_n^2 I \right]^{-1} X \mathbf{y}$$, 
which would be the estimate
we would get with the standard [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression).
In fact this corresponds exactly to having weighted ridge regression with weights $$\Sigma_p^{-1}$$.


Next consider the case where we have a new data point $$\mathbf{x}_{\ast}$$. We want to predict
$$\mathbf{f}_{\ast} = \mathbf{x}_{\ast}^\top\mathbf{w}$$. We note that 
$$\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y}$$ is Gaussian because $$\mathbf{f}_{\ast}$$
is a linear combination of $$\mathbf{w}$$ and $$\mathbf{w} \mid  X, \mathbf{y}$$ is a Gaussian given by
eq. \eqref{eq:map-w}. 

Thus we have that,

$$
\begin{align}
\mathbb{E}[\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y}] &= \mathbb{E}[\mathbf{x}_{\ast}^\top\mathbf{w} \mid \mathbf{x}_{\ast}, X, \mathbf{y}], \nonumber \\
&\overset{(a)}{=} \mathbf{x}_{\ast}^\top \mathbb{E}[\mathbf{w} \mid \mathbf{x}_{\ast}, X, \mathbf{y}], \nonumber \\
&\overset{(b)}{=} \frac{1}{\sigma_n^2} \mathbf{x}_{\ast}^\top A^{-1} X \mathbf{y}, \nonumber
\end{align}
$$

where $(a)$ follows from linearity of expectation, and $(b)$ follows from $$\mathbf{w}$$ being independent of $$\mathbf{x}_{\ast}$$ when conditioned on $$\mathcal{D}$$, using the posterior mean from \eqref{eq:map-w}.
Similarly we have that

$$
\begin{align}
\text{Cov}[\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y}] = \text{Cov}[\mathbf{x}_{\ast}^\top\mathbf{w} \mid \mathbf{x}_{\ast}, X, \mathbf{y}] &= \mathbf{x}_{\ast}^\top  \text{Cov}[\mathbf{w} \mid X, \mathbf{y}]\mathbf{x}_{\ast}, \nonumber \\
&= \mathbf{x}_{\ast}^\top A^{-1}\mathbf{x}_{\ast}. \nonumber
\end{align}
$$

where we use the linearity in computing covariances of Gaussians. This gives us the predictive distribution as 

$$
\begin{align}
\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y} &\sim \mathcal{N}\left(
\frac{1}{\sigma_n^2} \mathbf{x}_{\ast}^\top A^{-1} X \mathbf{y}, \mathbf{x}_{\ast}^\top A^{-1}\mathbf{x}_{\ast}
\right)
\end{align}
$$

We note that the predictive mean here is basically a linear combination of the observed points $$\mathbf{y}$$.


### Closer look at the prediction

To take a closer look at the prediction let us start with some matrix algebra to simplify the predictive mean and covariance

$$
\begin{align*}
\frac{1}{\sigma_n^2} X \left[X^\top \Sigma_p X + \sigma_n^2 I\right] &= \frac{1}{\sigma_n^2} X X^\top \Sigma_p X + X,\\
&= \left[\frac{1}{\sigma_n^2} X X^\top \Sigma_p + I \right] X, \\
&= \left[\frac{1}{\sigma_n^2} X X^\top  + \Sigma_p^{-1} \right]\Sigma_p X,\\
&= A \Sigma_p X, \\
\implies A^{-1} \frac{1}{\sigma_n^2} X \left[X^\top \Sigma_p X + \sigma_n^2 I\right] &= \Sigma_p X, \\
\implies \frac{1}{\sigma_n^2} A^{-1} X \left[X^\top \Sigma_p X + \sigma_n^2 I\right] &= \Sigma_p X, \\
\implies \frac{1}{\sigma_n^2} A^{-1} X &= \Sigma_p X \left[X^\top \Sigma_p X + \sigma_n^2 I\right]^{-1}.
\end{align*}
$$

Using this to substitute for $$\frac{1}{\sigma_n^2} A^{-1} X$$ in the mean, we have that

$$
\begin{align*}
\mathbb{E}[\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y}] &= \mathbf{x}_{\ast}^\top \Sigma_p X \left[X^\top \Sigma_p X + \sigma_n^2 I\right]^{-1} \mathbf{y}
\end{align*}
$$

#### The Sherman-Morrisson-Woodbury Lemma

The Sherman-Morrisson-Woodbury lemma or the matrix-inversion-lemma is a result in matrix analysis that 
allows us to invert matrices. Given invertible matrices $$Z \in \mathbb{R}^{n \times n}$$, and 
$$W \in \mathbb{R}^{m \times m}$$, and an invertible matrix $$Z + XWY^\top$$, with matrices $$ X, Y \in \mathbb{R}^{n \times m}$$,
we have that

$$
\begin{align}
\left[ Z + XWY^\top \right]^{-1} &= Z^{-1} - Z^{-1}X\left[ W^{-1} + Y^\top Z^{-1} X \right] Y^{\top} Z^{-1}
\end{align}
$$


Using this with the matrix $$A$$, we have that,

$$
\begin{align*}
A^{-1} &= \left[\Sigma_p^{-1} + X \frac{1}{\sigma_n^2} I X^\top\right]^{-1},\\
&= \Sigma_p - \Sigma_p X \left[ \sigma_n^2 I + X^\top \Sigma_p X \right]^{-1} X^\top \Sigma_p, \\
&= \Sigma_p - \Sigma_p X \left[ \sigma_n^2 I + K \right]^{-1} X^\top \Sigma_p
\end{align*}
$$

where $$K= X^\top \Sigma_p X$$.

Substituting this in the covariance formula gives us that

$$
\begin{align*}
\text{Cov}[\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y}] &= \mathbf{x}_{\ast}^\top \Sigma_p \mathbf{x}_{\ast} - 
\mathbf{x}_{\ast}^\top \Sigma_p X \left[ \sigma_n^2 I + K \right]^{-1} X^\top \Sigma_p \mathbf{x}_{\ast}.
\end{align*}
$$

This gives us that

$$
\begin{align}
\mathbf{f}_{\ast} \mid \mathbf{x}_{\ast}, X, \mathbf{y} &\sim \mathcal{N}\left(
\mathbf{x}_{\ast}^\top \Sigma_p X \left[\sigma_n^2 I + K \right]^{-1} \mathbf{y}, 
\mathbf{x}_{\ast}^\top \Sigma_p X \left[ \sigma_n^2 I + K \right]^{-1} X^\top \Sigma_p \mathbf{x}_{\ast}
\right) \label{eq:final-predictive-distribution}
\end{align}
$$

This distribution is very interesting because it shows us that the distribution depends only on inner
products between the training points $$X$$ -- in the form of $$K$$, and between the test point $$\mathbf{x}_\ast$$ 
and the training points $$X$$, both in the spaces scaled by $$\Sigma_p$$, without actually depending on the individual
training points. The mean depends on the training target points $$\mathbf{y}$$. This observation will
be important as we proceed further.

### Projection of inputs into a larger feature space

The Bayesian linear model we have worked with till now has limited expressiveness. One way to overcome this
is to project the input covariates into a much higher dimensional space and fit the linear model there. For 
example, we could project a scalar input $$x$$ into a space of the powers of $$x$$, i.e. $$(1, x, x^2, x^3, \cdots)^\top$$
to regress on all polynomials on the input. The model trained is still linear in the weights and thus tractable.

More formally, we define $$\phi: \mathbb{R}^D \rightarrow \mathbb{R}^N$$, where $$N$$ is much larger than $$D$$ and do 
inference in that place. We define 

$$
\begin{align}
f (\mathbf{x}) &= \phi(\mathbf{x})^\top \mathbf{w}, \\
y &= f (\mathbf{x}) + \epsilon,
\end{align}
$$

Where we have $$\epsilon \sim \mathcal{N}(0, \sigma_n^2)$$ as before and 
$$\mathbf{w} \sim \mathcal{N}(0, \Sigma_p)$$, where now we have $$\Sigma_p \in \mathbb{R}^{N \times N}$$
(rather than $$\mathbb{R}^{D \times D} $$).

We use the same vectorisation as before and use $$\phi(X) = \Phi$$ to represent transformations on all the
training data points. For a new test point $$\mathbf{x}_{\ast}$$, let us represent its projection by
$$\phi_{\ast} = \phi(\mathbf{x}_{\ast})$$. Further let $$\mathbf{f}_\ast$$ be the predicted value for
$$\mathbf{x}_{\ast}$$. From eq. \eqref{eq:final-predictive-distribution}, it follows that,

$$
\begin{align}
\mathbb{E}[\mathbf{f}_{\ast} \mid \phi_\ast, \Phi, \mathbf{y}] &= \phi_\ast^\top \Sigma_p \Phi \left[\sigma_n^2 I + K \right]^{-1} \mathbf{y}, \nonumber\\
\text{Cov}[\mathbf{f}_{\ast} \mid \phi_\ast, \Phi, \mathbf{y}] &= \phi_\ast^\top \Sigma_p \Phi \left[ \sigma_n^2 I + K \right]^{-1} \Phi^\top \Sigma_p \phi_\ast, \nonumber
\end{align}
$$

where $$K=\Phi^\top \Sigma_p \Phi$$ now.

As before we note that this depends only on the inner products of the projection into $$\mathbb{R}^N$$
between the training points and the training and test points; and not the projections themselves. 

Given two points $$\mathbf{x}$$ and $$\mathbf{x}'$$, we can define their covariance function or kernel as 

$$
\begin{align}
k(\mathbf{x}, \mathbf{x}') &= \phi(\mathbf{x})^\top \Sigma_p \phi(\mathbf{x}'),\\
&\overset{(a)}{=} \phi(\mathbf{x})^\top \Sigma_p^{\frac{1}{2}} \Sigma_p^{\frac{1}{2}} \phi(\mathbf{x}'), \nonumber\\
&= \psi(\mathbf{x})^\top \psi(\mathbf{x}') \nonumber
\end{align}
$$

where $(a)$ follows because the covariance matrix $$\Sigma_p$$ admits a square root: it is symmetric positive definite.
and $$\psi(\mathbf{x}) = \Sigma_p^{\frac{1}{2}} \phi(\mathbf{x})$$.

We can then write the predictive distribution as

$$
\begin{align}
\mathbb{E}[\mathbf{f}_{\ast} \mid \phi_\ast, \Phi, \mathbf{y}] &= k(\mathbf{x}_{\ast}, X) \left[\sigma_n^2 I + k(X,X) \right]^{-1} \mathbf{y}, \nonumber\\
\text{Cov}[\mathbf{f}_{\ast} \mid \phi_\ast, \Phi, \mathbf{y}] &= k(\mathbf{x}_{\ast}, X) \left[ \sigma_n^2 I + k(X,X) \right]^{-1} k(X, \mathbf{x}_{\ast}), \nonumber
\end{align}
$$

overloading $$k$$ to work with vectors as well. 
The key observation is that the predictive distribution 
we derived above depends on the data only through the kernel
evaluations $$k(\mathbf{x}_i, \mathbf{x}_j)$$ between training points, and $$k(\mathbf{x}_i, \mathbf{x}_{\ast})$$ between
training points and the test point -- the projections $$\phi(\cdot)$$ themselves never appear. This means we can work
directly with a kernel $$k(\cdot, \cdot)$$ without ever constructing the feature map. 
This is called the _kernel trick_.
For example, with the identity
feature map $$\phi(\mathbf{x}) = \mathbf{x}$$ we recover the linear kernel 
$$k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \Sigma_p \mathbf{x}'$$
from the first half of the post. But nothing stops us from using other kernels, which might be finite dimensional or even
infinite dimensional as we shall see.

## Feature-Space View

An alternative view of our development in the previous section would be to view Gaussian Processes
as fitting functions. To begin, we define a _Gaussian Process_.

**Gaussian Process:** A Gaussian process is a collection of random variables, any finite number
of which are jointly Gaussian distributed.

Since Gaussians are completely specified by their mean and covariance, a Gaussian process is
completely specified by its mean and covariance function. That is

$$
\begin{align*}
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')),
\end{align*}
$$

if 

$$
\begin{align}
m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})], \\
k(\mathbf{x}, \mathbf{x}') &= \mathbb{E}[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x}')-m(\mathbf{x}')) ]
\end{align}
$$

**Example:** A simple example of a Gaussian process is 

$$
\begin{align*}
f(\mathbf{x}) = \phi(\mathbf{x})^\top \mathbf{w},
\end{align*}
$$

for $$\mathbf{x} \in \mathbb{R}^{D}$$, $$\phi : \mathbb{R}^{D} \rightarrow \mathbb{R}^{N}$$, and
$$\mathbf{w} \sim \mathcal{N}(0, \Sigma_p)$$, for some covariance matrix $$\Sigma_p \in \mathbb{R}^{N \times N}$$.
For this setup, we end up having that,

$$
\begin{align*}
m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})] = \mathbf{0}, \\
k(\mathbf{x}, \mathbf{x}') &= \mathbb{E}[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x}')-m(\mathbf{x}')) ] = \phi(\mathbf{x})^\top\Sigma_p\phi(\mathbf{x}),
\end{align*}
$$

We note that, for points $$\mathbf{x}_1, \cdots, \mathbf{x}_n$$, written compactly as a matrix $$X \in \mathbb{R}^{D \times n}$$,
we have that $$f(X) \sim \mathcal{N}(0, \phi(X)^\top \Sigma_p \phi(X))$$, overloading $$f$$ and $$\phi$$ to work in the 
vector case in the expected manner. We note that if $$n > N$$, the covariance matrix has 
a rank of at most $$N < n$$, and thus the Gaussian distribution is degenerate.

We note that $$\mathbf{x}$$ is often used for time in modelling, but need not be. In some modelling
problems, it is actually a high dimensional variable. 

One commonly used covariance function is the _squared exponential_, which is

$$
\begin{align}
k (\mathbf{x}_p, \mathbf{x}_q) &= \sigma_f^2 \exp\left( -\frac{1}{2 \ell^2} \|\mathbf{x}_p-\mathbf{x}_q\|_2^2 \right),
\end{align}
$$

where $$\sigma_f$$ and $$\ell$$ are hyperparameters. We note that this covariance is $$\sigma_f^2$$ when
$$\mathbf{x}_p=\mathbf{x}_q$$, making $$\sigma_f^2$$ to be the signal variance. The covariance decays as the
distance between $$\mathbf{x}_p$$ and $$\mathbf{x}_q$$ increases, but that is determined by the length scale 
$$\ell$$.

For most of the next discussion we assume $$m(\mathbf{x})=\mathbf{0}$$.

### Prediction with noise-free observations

We first consider the case where we have noiseless observations. That is we observe 
$$n$$ points: 
$$\{ (\mathbf{x}_i, f_i) \mid i = 1, 2, \cdots, n \}$$. Let us be interested in 
predicting the value at $$n_*$$ points -- $$\mathbf{x}_{i*}, 1 \le i \le n_*$$. 
We want to predict $$\mathbf{f}_* = (f_{1*}, f_{2*}, \cdots, f_{n_*})$$.

We have that 

$$
\begin{align}
\begin{bmatrix} \mathbf{f}, \\ \mathbf{f}_* \end{bmatrix} &\sim \mathcal{N} \left(\mathbf{0}, 
 \begin{bmatrix} K(X, X) \quad K(X, X_*) \\ K(X_*, X) \quad K(X_*, X_*) \end{bmatrix} \right),
\end{align}
$$

where $$K(X, X_*)$$ is the $$n \times n_*$$ matrix of covariance between the $$n$$
training points and $$n_*$$ test points; and similarly $$K(X, X)$$,
$$K(X_*, X)$$ and $$K(X_*, X_*)$$. From eq. \eqref{eq:final-predictive-distribution}, 
we have that, 

$$ 
\begin{align}
\mathbf{f}_* | X_*, X, \mathbf{f} &\sim \mathcal{N} \left( K(X_*, X)K(X, X)^{-1}\mathbf{f}, K(X_*, X_*) - K(X_*, X)K(X, X)^{-1}K(X, X_*)  \right)
\end{align}
$$

### Prediction from noisy observations

We model noisy observations by modelling the covariance matrix as 

$$
\begin{align}
\text{Cov}(y_p, y_q) &= k(\mathbf{x}_p, \mathbf{x}_q) + \sigma_n^2\delta_{p,q},\\
\text{Cov}(\mathbf{y}) &= K(X, X) + \sigma_n^2 I
\end{align}
$$

This gives us that

$$
\begin{align}
\begin{bmatrix} \mathbf{y}, \\ \mathbf{f}_* \end{bmatrix} &\sim \mathcal{N} \left(\mathbf{0}, 
 \begin{bmatrix} K(X, X) + \sigma_n^2 I & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{bmatrix} \right),
\end{align}
$$

which gives us that 

$$ 
\begin{align}
\mathbf{f}_* | X_*, X, \mathbf{y} &\sim \mathcal{N} \left( K(X_*, X)\left[K(X, X) + \sigma_n^2 I \right]^{-1}\mathbf{y}, K(X_*, X_*) - K(X_*, X)\left[K(X, X) + \sigma_n^2 I \right]^{-1}K(X, X_*)  \right)
\end{align}
$$

In the special case where we are only predicting for one point $$\mathbf{x}_*$$, writing
$$\mathbf{k}_* = k(\mathbf{x}_*) = K(X,\mathbf{x}_*)$$, we have that the predicted value is

$$
\begin{align}
\hat{f}(\mathbf{x}_*) &= \mathbf{k}_*^\top\left[K(X, X) + \sigma_n^2 I \right]^{-1}\mathbf{y},\\
&= \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \mathbf{x}_*),
\end{align}
$$

where $$\mathbf{\alpha}=\left[K(X, X) + \sigma_n^2 I \right]^{-1}\mathbf{y}$$. This gives us that 
the predicted value for a test point is a weighted linear combination of the inner products of the 
test point and each training point. 

It is easy to see that the variance of the predictor is

$$
\begin{align*}
\text{Var}\left[\hat{f}(\mathbf{x}_*)\right] = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top\left[K(X, X) + \sigma_n^2 I \right]^{-1}\mathbf{k}_*.
\end{align*}
$$

