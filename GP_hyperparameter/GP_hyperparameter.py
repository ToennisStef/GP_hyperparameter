import numpy as np 
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.gp_hyperparameter import log_marginal_likelihood

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='MLL Visualization', 
            style={'textAlign': 'center'}),
    html.Div(children='Visualization of the log marginal likelihood of a Gaussian Process with respect to the hyperparameters.',
             style={'textAlign': 'center'}),
    dcc.Graph(id='mll-plot'),
    html.Div([
        html.Label('x1'),
        dcc.Slider(id='x1-slider', min=0, max=5, value=1),
        html.Label('x2'),
        dcc.Slider(id='x2-slider', min=0, max=5, value=1.8),
        html.Label('y1'),
        dcc.Slider(id='y1-slider', min=0, max=5, value=2),
        html.Label('y2'),
        dcc.Slider(id='y2-slider', min=0, max=5, value=2.5)
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
    