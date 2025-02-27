import sympy
import numpy as np
import GPy

def log_marginal_likelihood():
    # Define the length scale symbol
    l = sympy.symbols(r"\theta_l")
    x1 = sympy.symbols(r"x_1")
    x2 = sympy.symbols(r"x_2")
    y1 = sympy.symbols(r"y_1")
    y2 = sympy.symbols(r"y_2")
    mu1 = sympy.symbols(r"\mu_1")
    mu2 = sympy.symbols(r"\mu_2")
    sigma = sympy.symbols(r"\sigma")
    sigma_n = sympy.symbols(r"\sigma_n")

    # Define the kernel function
    squared_exponential = lambda x,y: sympy.exp(- ((x-y)**2)/ (2*l))

    x = sympy.Matrix([x1, x2])
    y = sympy.Matrix([y1, y2])
    mu = sympy.Matrix([mu1, mu2])

    Sigma = [[squared_exponential(xi, xj) for xj in x] for xi in x]
    Sigma = sympy.Matrix(Sigma)

    N = sympy.Matrix([[sigma_n, 0], [0, sigma_n]])

    Cov = Sigma + N
    Cov_inv = Cov.inv()
    Cov_log_det_m = sympy.Matrix([sympy.log(Cov.det())])
    mll = (-1)* (1/2 * (mu - y).T * Cov_inv * (mu - y) + 1/2 * Cov_log_det_m + 1/2 * 2 * sympy.Matrix([sympy.log(2*sympy.pi)]))

    mll_func = sympy.lambdify((x1, x2, y1, y2, mu1, mu2, sigma, sigma_n, l), mll)
    
    return mll_func

def calc_GP(noise_inference: str, x: list, y: list):
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y).reshape(-1, 1)
    
    # Create a kernel
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

    # Create a Gaussian Process model
    model = GPy.models.GPRegression(X, Y, kernel)

    # Set the noise variance to a very small value and fix it
    if noise_inference == 'fixed zero':
        model.Gaussian_noise.variance = 1e-6
        model.Gaussian_noise.variance.fix()
    
    # Optimize the model
    model.optimize()
    model.optimize_restarts(num_restarts=10, verbose=False)
    
    # Make predictions
    X_plot = np.linspace(-5, 5, 200).reshape(-1, 1)
    Y_mean, Y_var = model.predict(X_plot)
    
    return X_plot.flatten().tolist(), Y_mean.flatten().tolist(), Y_var.flatten().tolist()