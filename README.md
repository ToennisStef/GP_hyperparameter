# GP_hyperparameter

Visualisation of GP training an marginal log likelihood


## Evaluating the influence of noise on the lengthscale parameter during training</h1>

training is done by maximizing the marginal log likelihood (mll): 

$\log(y|\mathbf{x},\theta) = \frac{1}{2}[(\mathbf{y}-\mathbf{\mu})^\top](\Sigma+N)^{-1}(\mathbf{y}-\mathbf{\mu})+\log |{\Sigma+N}|+n\log 2\pi$

with:
- $\mathbf{y}=\begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}$: observations of the objective function
- $\mathbf{x}=\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}$: corresponding observation location
- $\mathbf{\mu}=\begin{pmatrix} \mu(x_1) \\ \vdots \\ \mu(x_n) \end{pmatrix}=\begin{pmatrix} \theta_\mu \\ \vdots \\ \theta_\mu \end{pmatrix}$: prior mean of the GP at the corresponding observation location is usally a constant, and derived through model training
- $\Sigma=[k(x,x')]_{\forall x,x' \in \mathbf{x}}$: the covariance matrix of the GP with respect to the observation locations.
- $k(x,x')$: is the covariance function
- $N=\sigma_{n}I = \begin{pmatrix} \sigma_{n} & 0 \\ 0 & \sigma_{n} \end{pmatrix}$: Noise Covariance matrix 
- $n$: is the number of available observations

I want to derive the derivative of the mll with respect to the lengthscale!
with the mll being:

Let's assume therefore homeoskedastic noise in the form:

$N=\sigma_{n}I = \begin{pmatrix}
\sigma_{n} & 0 \\
0 & \sigma_{n}
\end{pmatrix}$

, lets also assume two observations:

$\mathbf{x}=\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$, $\mathbf{y}=\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}$, $\mathbf{\mu}=\begin{pmatrix} \theta_\mu \\ \theta_\mu \end{pmatrix}$

with the squared exponential covariance function:

$k(x,x')=\sigma*\exp(\frac{-(x-x')^2}{2*\theta^2})$

The Covariance matrix becomes:

$\Sigma = \begin{pmatrix}
1 & \sigma\exp\left(-\frac{(x_1 - x_2)^2}{2\theta_l}\right) \\
\sigma\exp\left(-\frac{(x_1 - x_2)^2}{2\theta_l}\right) & 1
\end{pmatrix}$

$\Sigma+N= \begin{pmatrix}
\sigma_n + 1 & \sigma\exp\left(-\frac{(x_1 - x_2)^2}{2\theta_l}\right) \\
\sigma\exp\left(-\frac{(x_1 - x_2)^2}{2\theta_l}\right) & \sigma_n + 1
\end{pmatrix}$

Intermediate result: 
- Assuming noisy meassurements is beneficial for numerical stability as it can be seen as regularization term for the covariance matrix. The calculation of the matrix inversion becomes therefore more stable. 

I am using sympy for symbolic evaluation of the equation automatic differentiation.
Deriving an analytic form of the mll is nontrivial and results in an rather complicated expression