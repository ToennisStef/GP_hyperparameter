import numpy as np 
import sympy
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

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

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='mll-plot'),
    html.Div([
        html.Label('x1'),
        dcc.Slider(id='x1-slider', min=0, max=5, step=0.1, value=1),
        html.Label('x2'),
        dcc.Slider(id='x2-slider', min=0, max=5, step=0.1, value=1.8),
        html.Label('y1'),
        dcc.Slider(id='y1-slider', min=0, max=5, step=0.1, value=2),
        html.Label('y2'),
        dcc.Slider(id='y2-slider', min=0, max=5, step=0.1, value=2.5)
    ])
])

@app.callback(
    Output('mll-plot', 'figure'),
    [Input('x1-slider', 'value'),
     Input('x2-slider', 'value'),
     Input('y1-slider', 'value'),
     Input('y2-slider', 'value')]
)
def update_plot(x1, x2, y1, y2):
    mll_func = log_marginal_likelihood()
    
    # Define resolution of the meshgrid
    res = 100

    # Define the range of the hyperparameters
    l_upper = 150
    l_lower = 0.01

    sigma_n_upper = 1e-1
    sigma_n_lower = 1e-6

    mu1 = 3
    mu2 = mu1
    sigma = 0.2
    sigma_n = np.linspace(sigma_n_lower, sigma_n_upper, res)
    l = np.linspace(l_lower, l_upper, res)

    sigma_n_MG, l_MG = np.meshgrid(sigma_n, l)

    mll_vals = mll_func(x1, x2, y1, y2, mu1, mu2, sigma, sigma_n_MG, l_MG)

    mll_vals_ = mll_vals[0, 0, :, :]
    max_mll_fixed_l = np.max(mll_vals_, axis=1)
    max_mll_idx_fixed_l = np.argmax(mll_vals_, axis=1)

    max_mll_fixed_sigma = np.max(mll_vals_, axis=0)
    max_mll_idx_fixed_sigma = np.argmax(mll_vals_, axis=0)

    # Get the corresponding sigma_n values for the maximal mll values
    max_sigma_n = []
    for i, id in enumerate(max_mll_idx_fixed_l):
        max_sigma_n.append(sigma_n_MG[i, id])
    max_sigma_n = np.array(max_sigma_n)

    max_l = []
    for i, id in enumerate(max_mll_idx_fixed_sigma):
        max_l.append(l_MG[id, i])
    max_l = np.array(max_l)
    
    # Create figure
    fig = go.Figure()

    # Add surface plot
    fig.add_trace(go.Surface(z=mll_vals[0, 0, :, :], x=sigma_n_MG, y=l_MG, colorscale='viridis'))
    # Add 3D line plot for max_mll
    fig.add_trace(go.Scatter3d(
        x=max_sigma_n,
        y=l,
        z=max_mll_fixed_l,
        mode='lines',
        line=dict(color='red', width=5),
        name='Max MLL with fixed θₗ'
    ))

    # Add 3D line plot for max_mll
    fig.add_trace(go.Scatter3d(
        x=sigma_n,
        y=max_l,
        z=max_mll_fixed_sigma,
        mode='lines',
        line=dict(color='blue', width=5),
        name='Max MLL with fixed σₙ'
    ))

    # Set axis labels
    fig.update_layout(
        title="Interactive 3D Surface Plot",
        scene=dict(
            xaxis_title="σₙ",
            yaxis_title="θₗ",
            zaxis_title=dict(text="mll")
        )
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
# import numpy as np 
# import sympy
# import plotly.graph_objects as go
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from ipywidgets import interact, FloatSlider
# import ipywidgets as widgets

# import plotly.offline as pyo

# def log_marginal_likelihood():
#     # Define the length scale symbol
#     l = sympy.symbols(r"\theta_l")
#     x1 = sympy.symbols(r"x_1")
#     x2 = sympy.symbols(r"x_2")
#     y1 = sympy.symbols(r"y_1")
#     y2 = sympy.symbols(r"y_2")
#     mu1 = sympy.symbols(r"\mu_1")
#     mu2 = sympy.symbols(r"\mu_2")
#     sigma = sympy.symbols(r"\sigma")
#     sigma_n = sympy.symbols(r"\sigma_n")

#     # Define the kernel function
#     squared_exponential = lambda x,y: sympy.exp(- ((x-y)**2)/ (2*l))

#     x = sympy.Matrix([x1, x2])
#     y = sympy.Matrix([y1, y2])
#     mu = sympy.Matrix([mu1, mu2])

#     Sigma = [[squared_exponential(xi, xj) for xj in x] for xi in x]
#     Sigma = sympy.Matrix(Sigma)

#     N = sympy.Matrix([[sigma_n, 0], [0, sigma_n]])

#     Cov = Sigma + N
#     Cov_inv = Cov.inv()
#     Cov_log_det_m = sympy.Matrix([sympy.log(Cov.det())])
#     mll = (-1)* (1/2 * (mu - y).T * Cov_inv * (mu - y) + 1/2 * Cov_log_det_m + 1/2 * 2 * sympy.Matrix([sympy.log(2*sympy.pi)]))

#     mll_func = sympy.lambdify((x1, x2, y1, y2, mu1, mu2, sigma, sigma_n, l), mll)
    
#     return mll_func

# def update_plot(x1, x2, y1, y2):
#     mll_func = log_marginal_likelihood()
    
#     # Define resolution of the meshgrid
#     res = 100

#     # Define the range of the hyperparameters
#     l_upper = 150
#     l_lower = 0.01

#     sigma_n_upper = 1e-1
#     sigma_n_lower = 1e-6

#     mu1 = 3
#     mu2 = mu1
#     sigma = 0.2
#     sigma_n = np.linspace(sigma_n_lower, sigma_n_upper, res)
#     l = np.linspace(l_lower, l_upper, res)

#     sigma_n_MG, l_MG = np.meshgrid(sigma_n, l)

#     mll_vals = mll_func(x1, x2, y1, y2, mu1, mu2, sigma, sigma_n_MG, l_MG)

#     mll_vals_ = mll_vals[0, 0, :, :]
#     max_mll_fixed_l = np.max(mll_vals_, axis=1)
#     max_mll_idx_fixed_l = np.argmax(mll_vals_, axis=1)

#     max_mll_fixed_sigma = np.max(mll_vals_, axis=0)
#     max_mll_idx_fixed_sigma = np.argmax(mll_vals_, axis=0)

#     # Get the corresponding sigma_n values for the maximal mll values
#     max_sigma_n = []
#     for i, id in enumerate(max_mll_idx_fixed_l):
#         max_sigma_n.append(sigma_n_MG[i, id])
#     max_sigma_n = np.array(max_sigma_n)

#     max_l = []
#     for i, id in enumerate(max_mll_idx_fixed_sigma):
#         max_l.append(l_MG[id, i])
#     max_l = np.array(max_l)
    
#     # Update the plot
#     with fig.batch_update():
#         fig.data[0].z = mll_vals[0, 0, :, :]
#         fig.data[0].x = sigma_n_MG
#         fig.data[0].y = l_MG
#         fig.data[1].x = max_sigma_n
#         fig.data[1].y = l
#         fig.data[1].z = max_mll_fixed_l
#         fig.data[2].x = sigma_n
#         fig.data[2].y = max_l
#         fig.data[2].z = max_mll_fixed_sigma

# # Create figure
# fig = go.Figure()

# # Add surface plot
# fig.add_trace(go.Surface(colorscale='viridis'))
# # Add 3D line plot for max_mll
# fig.add_trace(go.Scatter3d(
#     mode='lines',
#     line=dict(color='red', width=5),
#     name='Max MLL with fixed θₗ'
# ))

# # Add 3D line plot for max_mll
# fig.add_trace(go.Scatter3d(
#     mode='lines',
#     line=dict(color='blue', width=5),
#     name='Max MLL with fixed σₙ'
# ))

# # Set axis labels
# fig.update_layout(
#     title="Interactive 3D Surface Plot",
#     scene=dict(
#         xaxis_title="σₙ",
#         yaxis_title="θₗ",
#         zaxis_title=dict(text="mll")
#     )
# )

# # Create interactive sliders
# interact(update_plot,
#          x1=FloatSlider(min=0, max=5, step=0.1, value=1, description='x1'),
#          x2=FloatSlider(min=0, max=5, step=0.1, value=1.8, description='x2'),
#          y1=FloatSlider(min=0, max=5, step=0.1, value=2, description='y1'),
#          y2=FloatSlider(min=0, max=5, step=0.1, value=2.5, description='y2')
#         )

# # Display the figure
# # fig.show()
# pyo.plot(fig, filename="mll_surface_plot.html")
# fig.write_html("mll_surface_plot.html")

# # import numpy as np 
# # import sympy
# # import plotly.graph_objects as go


# # def log_marginal_likelihood():
    
# #     # Define the length scale symbol
# #     l = sympy.symbols(r"\theta_l")
# #     x1 = sympy.symbols(r"x_1")
# #     x2 = sympy.symbols(r"x_2")
# #     y1 = sympy.symbols(r"y_1")
# #     y2 = sympy.symbols(r"y_2")
# #     mu1 = sympy.symbols(r"\mu_1")
# #     mu2 = sympy.symbols(r"\mu_2")
# #     sigma = sympy.symbols(r"\sigma")
# #     sigma_n = sympy.symbols(r"\sigma_n")

# #     # Define the kernel function
# #     squared_exponential = lambda x,y: sympy.exp(- ((x-y)**2)/ (2*l))

# #     x = sympy.Matrix([x1, x2])
# #     y = sympy.Matrix([y1, y2])
# #     mu = sympy.Matrix([mu1, mu2])

# #     Sigma = [[squared_exponential(xi, xj) for xj in x] for xi in x]
# #     Sigma = sympy.Matrix(Sigma)

# #     N = sympy.Matrix([[sigma_n, 0], [0, sigma_n]])

# #     Cov = Sigma + N
# #     Cov_inv = Cov.inv()
# #     Cov_log_det_m = sympy.Matrix([sympy.log(Cov.det())])
# #     mll = (-1)* (1/2 * (mu - y).T * Cov_inv * (mu - y) + 1/2 * Cov_log_det_m + 1/2 * 2 * sympy.Matrix([sympy.log(2*sympy.pi)]))

# #     mll_func = sympy.lambdify((x1, x2, y1, y2, mu1, mu2, sigma, sigma_n, l), mll)
    
# #     return mll_func

# # def main(mll_func:callable, 
# #             x1:float=1,
# #             x2:float=1.8,
# #             y1:float=2,
# #             y2:float=2.5,
# #             mu1:float=3,
# #             mu2:float=3,
# #             sigma:float=0.2,
# #             )->np.ndarray:
    
# #     # Define reolution of the meshgrid
# #     res = 100

# #     # Define the range of the hyperparameters
# #     l_upper = 150
# #     l_lower = 0.01

# #     sigma_n_upper = 1e-1
# #     sigma_n_lower = 1e-6

# #     x1 = 1
# #     x2 = 1.8
# #     y1 = 2
# #     y2 = 2.5
# #     mu1 = 3
# #     mu2 = mu1
# #     sigma = 0.2
# #     sigma_n = np.linspace(sigma_n_lower, sigma_n_upper,res)
# #     l = np.linspace(l_lower, l_upper, res)

# #     sigma_n_MG, l_MG = np.meshgrid(sigma_n, l)

# #     mll_vals = mll_func(x1, x2, y1, y2, mu1, mu2, sigma, sigma_n_MG, l_MG)

# #     mll_vals_ = mll_vals[0,0,:,:]
# #     max_mll_fixed_l = np.max(mll_vals_, axis=1)
# #     max_mll_idx_fixed_l = np.argmax(mll_vals_, axis=1)

# #     max_mll_fixed_sigma = np.max(mll_vals_, axis=0)
# #     max_mll_idx_fixed_sigma = np.argmax(mll_vals_, axis=0)


# #     # Get the corresponding sigma_n values for the maximal mll values
# #     max_sigma_n = []
# #     for i, id in enumerate(max_mll_idx_fixed_l):
# #         max_sigma_n.append(sigma_n_MG[i,id])
# #     max_sigma_n = np.array(max_sigma_n)

# #     max_l = []
# #     for i, id in enumerate(max_mll_idx_fixed_sigma):
# #         max_l.append(l_MG[id,i])
# #     max_l = np.array(max_l)
    
# #     #Plot the results    
# #     # Create figure
# #     fig = go.Figure()

# #     # Add surface plot
# #     fig.add_trace(go.Surface(z=mll_vals[0, 0, :, :], x=sigma_n_MG, y=l_MG, colorscale='viridis'))
# #     # Add 3D line plot for max_mll
# #     fig.add_trace(go.Scatter3d(
# #         x=max_sigma_n,
# #         y=l,
# #         z=max_mll_fixed_l,
# #         mode='lines',
# #         line=dict(color='red', width=5),
# #         name='Max MLL with fixed θₗ'
# #     ))

# #     # Add 3D line plot for max_mll
# #     fig.add_trace(go.Scatter3d(
# #         x=sigma_n,
# #         y=max_l,
# #         z=max_mll_fixed_sigma,
# #         mode='lines',
# #         line=dict(color='blue', width=5),
# #         name='Max MLL with fixed σₙ'
# #     ))

# #     # Set axis labels
# #     fig.update_layout(
# #         title="Interactive 3D Surface Plot",
# #         scene=dict(
# #             xaxis_title="σₙ",
# #             yaxis_title="θₗ",
# #             zaxis_title=dict(text="mll")
# #         )
# #     )

# #     # Add annotations for the values of x1_, x2_, y1_, y2_, mu1_, mu2_, sigma_, and res
# #     fig.add_annotation(
# #         text=f"x1={x1}, x2={x2}, y1={y1}, y2={y2}, μ1={mu1}, μ2={mu2}, σ={sigma}",
# #         xref="paper", yref="paper",
# #         x=0.5, y=-0.1,
# #         showarrow=False
# #     )

# #     # Save figure to HTML file
# #     fig.write_html("mll_surface_plot.html")
# #     print("Figure saved as mll_surface_plot.html")
        
# # mll_func = log_marginal_likelihood()
# # main(mll_func)