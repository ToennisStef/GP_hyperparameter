import json
from textwrap import dedent as d
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from src.gp_hyperparameter.updates import update_global_points, update_GP_fig
from src.gp_hyperparameter.funcs import calc_GP, forrester
from src.gp_hyperparameter.classes import DataPoints


external_stylesheets = ["https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Initial values for points
N_points = 2

DataPoints = [DataPoints() for _ in range(N_points)]

DataPoints[0].x = 0.1
DataPoints[0].y = forrester(DataPoints[0].x)
DataPoints[1].x = 0.675
DataPoints[1].y = forrester(DataPoints[1].x)

styles = {'pre': {'border': 'thin lightgrey solid', 'overflowX': 'scroll'}}
text_style = {'textAlign': 'center', "fontFamily": "Roboto, sans-serif"}
description_style = {'textAlign': 'left', "fontFamily": "Roboto, sans-serif"}

app.layout = html.Div([
    # Header
    html.H1(children='MLL Visualization', 
            style=text_style),
    # Description
    # html.Div(children='Visualization of the log marginal likelihood of a Gaussian Process with respect to the hyperparameters. The plot shows the log marginal likelihood as a function of the hyperparameters σₙ and θₗ. The red line shows the maximum log marginal likelihood for fixed θₗ and the blue line shows the maximum log marginal likelihood for fixed σₙ. The plot updates as the sliders are adjusted. You can adjust the position of the observation and the observation value',
    #          style=text_style),
    # Plots (GP-plot and MLL-Plot)
    html.Div([
        dcc.Graph(id='GP-plot', 
                config={'editable': True, 'edits': {'shapePosition': True}},
                style={'width': '45vw', 'aspect-ratio': '1/1'}),
        dcc.Graph(id='MLL-plot',
                style={'width': '45vw', 'aspect-ratio': '1/1'})
        ],
        style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'gap': '20px'}
    ),
    # Noise inference dropdown
    dcc.Dropdown(id='noise_inference', 
                 options=[
                     {'label': 'Fixed Zero Noise', 'value': 'fixed zero'},
                     {'label': 'Inferred Noise', 'value': 'inferred'}
                 ],
                 value='inferred'),
    # Display current hyperparameters
    html.Div([
        dcc.Markdown(id='hyperparameter-display', 
                     style=description_style),
            ], 
        style={'textAlign': 'center', 'marginTop': '20px'}
    ),
    html.Div([
        html.H2('Description', style=text_style),
        dcc.Markdown(
            r"""
            This is a Visualization of the log marginal likelihood of a Gaussian Process with respect to the hyperparameters. 
            The left plot shows a Gaussian Process fitted to the datapoints. 
            The right plot shows the log marginal likelihood of the Gaussian Process with respect as a function of the hyperparameters lengthscale ${\theta_l}$ and noise variance of the observations $\sigma_{n}$.
            
            The Gaussian Process assumes a squared exponential kernel function. The squared exponential kernel function is given by:
            
            $$k(x,x') = \theta_{\sigma}\exp\left(-\frac{(x-x')^2}{2\theta_l}\right)$$
            
            with:
            - $\theta_{\sigma}$: the variance of the kernel function
            - $\theta_l$: the lengthscale of the kernel function
            - $x,x'$: the input variables
            
            The log marginal likelihood is given by:
            
            $$\log(y|\mathbf{x},\theta) = \frac{1}{2}[(\mathbf{y}-\mathbf{\mu})^\top](\Sigma+N)^{-1}(\mathbf{y}-\mathbf{\mu})+\log |{\Sigma+N}|+n\log 2\pi$$

            with:
            - $\mathbf{y}=\begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}$: observations of the objective function
            - $\mathbf{x}=\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}$: corresponding observation location
            - $\mathbf{\mu}=\begin{pmatrix} \mu(x_1) \\ \vdots \\ \mu(x_n) \end{pmatrix}=\begin{pmatrix} \theta_\mu \\ \vdots \\ \theta_\mu \end{pmatrix}$: prior mean of the GP at the corresponding observation location is usually a constant, and derived through model training
            - $\Sigma=[k(x,x')]_{\forall x,x' \in \mathbf{x}}$: the covariance matrix of the GP with respect to the observation locations.
            - $k(x,x')$: is the covariance function
            - $N=\sigma_{n}I = \begin{pmatrix} \sigma_{n} & 0 \\ 0 & \sigma_{n} \end{pmatrix}$: Noise Covariance matrix 
            - $n$: is the number of available observations
            """,
        mathjax=True,  # Enables LaTeX rendering
        style=description_style)
    ]),
    # Plotting the relayoutData
    # html.Div(
    #     className='six columns',
    #     children=[html.Div([
    #                 dcc.Markdown(
    #                     d("""**Zoom and Relayout Data**""")),
    #                 html.Pre(id='relayout-data', style=styles['pre']),
    #             ])]
    # ),
    dcc.Store(id='GP')
])

@app.callback(
    [Output('GP-plot', 'figure'),
    Output('GP', 'data'),
    Output('hyperparameter-display', 'children')],
    [Input('GP-plot', 'relayoutData'),
    Input('noise_inference', 'value')]
)
def update_GP_plot(relayoutData, noise_inference):
    """ Updates the scatter plot and captures new (x1, y1) and (x2, y2) positions. """
    
    update_global_points(relayoutData, DataPoints)
    
    x = [point.x for point in DataPoints]
    y = [point.y for point in DataPoints]
    
    model, X_plot, Y_mean, Y_var = calc_GP(noise_inference, x, y)
    
    Y_plot = [forrester(x) for x in X_plot]
    
    # Get hyperparameters
    lengthscale = model.kern.lengthscale.values[0]
    variance = model.kern.variance.values[0]
    noise_var = model.likelihood.variance.values[0]

    # Create hyperparameter display text
    hyperparameter_text = f"**Current Hyperparameters:**\n-  Lengthscale: {lengthscale:.4f}\n- Variance: {variance:.4f}\n- Noise Variance: {noise_var:.4f}"
    
    # Create scatter plot with draggable points
    fig = go.Figure()

    update_GP_fig(fig, x, y, X_plot, Y_plot, Y_mean, Y_var)
    
    # Set axis range
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[-10, 20])
    
    # Set axis titles
    fig.update_xaxes(title_text='X')
    fig.update_yaxes(title_text='Y')
    
    # Update legend position        
    fig.update_legends(
        x = 0.01,
        y = 0.99,
        traceorder='normal')
    
    return fig, model.to_dict(), hyperparameter_text

# Interactable output
# @app.callback(
#     Output('relayout-data', 'children'),
#     [Input('GP-plot', 'relayoutData')])
# def display_relayout_data(relayoutData):
#     return json.dumps(relayoutData, indent=2)


import GPy
import numpy as np
@app.callback(
    Output('MLL-plot', 'figure'),
    Input('GP','data')
)
def update_MLL_plot(GP_data):
    model = GPy.models.GPRegression.from_dict(GP_data)
    current_mll = model.log_likelihood().flatten().tolist()
    current_lengthscale = model.kern.lengthscale.values[0].flatten().tolist()  # Lengthscale of RBF kernel
    current_noise_variance = model.Gaussian_noise.variance.values[0].flatten().tolist()  # Noise variance
    
    res = 100
    
    # select reasonable range for the hyperparameters
    lengthscale = np.linspace(1e-6, 1, res)
    noise_variance = np.linspace(1e-6, 10, res)
    # lengthscale = np.linspace(1e-6, current_lengthscale[0]*2, res)
    # noise_variance = np.linspace(1e-6, current_noise_variance[0]*2, res)
    
    # Create a meshgrid
    L, N = np.meshgrid(lengthscale, noise_variance)
    mll_values = np.zeros(L.shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            model.kern.lengthscale = L[i, j]
            model.Gaussian_noise.variance = N[i, j]
            mll_values[i, j] = model.log_likelihood()
    
    max_mll_fixed_lengthscale = np.max(mll_values, axis=0)
    max_mll_idx_fixed_lengthscale = np.argmax(mll_values, axis=0)

    max_mll_fixed_noise_variance = np.max(mll_values, axis=1)
    max_mll_idx_fixed_noise_variance = np.argmax(mll_values, axis=1)

    
    # Get the corresponding sigma_n values for the maximal mll values
    max_noise_variance = []
    for i, id in enumerate(max_mll_idx_fixed_lengthscale):
        max_noise_variance.append(N[id,i])
    # max_noise_variance = np.array(max_noise_variance)

    max_lengthscale = []
    for i, id in enumerate(max_mll_idx_fixed_noise_variance):
        max_lengthscale.append(L[i,id])
    # max_lengthscale = np.array(max_lengthscale)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=lengthscale,
        y=noise_variance,
        z=mll_values,
        colorscale='Viridis',
        name='MLL Surface'
    ))
    
    fig.add_trace(go.Scatter3d(
    x=lengthscale,
    y=max_noise_variance,
    z=max_mll_fixed_lengthscale,
    mode='lines',
    line=dict(color='red', width=5),
    name='Optimal σₙ for fixed θₗ'
    ))

    # Add 3D line plot for max_mll
    fig.add_trace(go.Scatter3d(
        x=max_lengthscale,
        y=noise_variance,
        z=max_mll_fixed_noise_variance,
        mode='lines',
        line=dict(color='blue', width=5),
        name='Optimal θₗ for fixed σₙ'
    ))

    
    fig.add_trace(go.Scatter3d(
        x=current_lengthscale,
        y=current_noise_variance,
        z=current_mll,
        name='MLL of GP',
        showlegend=True,
        ))
    
    # Set axis titles
    fig.update_layout(
        scene=dict(
            xaxis_title='Lengthscale',
            yaxis_title='Noise Variance',
            zaxis_title='Log Marginal Likelihood'
        )
    )
    
    # Update legend position        
    fig.update_legends(
        x = 0.01,
        y = 0.99,
        traceorder='normal')
    
    return fig
    

if __name__ == '__main__':
    app.run_server(debug=True)