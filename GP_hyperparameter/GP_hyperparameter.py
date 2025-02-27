import numpy as np 
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.gp_hyperparameter.funcs import log_marginal_likelihood

app = Dash(__name__)

# Initial values for points
initial_points = {'x': [1, 4], 'y': [1, 4]}

app.layout = html.Div([
    # Header
    html.H1(children='MLL Visualization', 
            style={'textAlign': 'center'}),
    # Description
    html.Div(children='Visualization of the log marginal likelihood of a Gaussian Process with respect to the hyperparameters. The plot shows the log marginal likelihood as a function of the hyperparameters σₙ and θₗ. The red line shows the maximum log marginal likelihood for fixed θₗ and the blue line shows the maximum log marginal likelihood for fixed σₙ. The plot updates as the sliders are adjusted. You can adjust the position of the observation and the observation value',
             style={'textAlign': 'center'}),
    dcc.Graph(id='mll-plot'),
    dcc.Graph(id='scatter-plot'),
])


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'relayoutData')
)
def update_scatter(relayout_data):
    """ Updates the scatter plot and captures new (x1, y1) and (x2, y2) positions. """
    
    global initial_points  # To keep track of dragged points

    # Check if user moved points
    if relayout_data and "shapes" in relayout_data:
        for i, shape in enumerate(relayout_data["shapes"]):
            initial_points['x'][i] = shape["x0"]
            initial_points['y'][i] = shape["y0"]

    # Create scatter plot with draggable points
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=initial_points['x'],
        y=initial_points['y'],
        mode='markers',
        marker=dict(size=10, color=['red', 'blue']),
        name="Draggable Points"
    ))

    # Add draggable shapes
    fig.update_layout(
        dragmode="drawopenpath",
        shapes=[
            dict(
                type="circle",
                xref="x", yref="y",
                x0=initial_points['x'][i] - 0.1, x1=initial_points['x'][i] + 0.1,
                y0=initial_points['y'][i] - 0.1, y1=initial_points['y'][i] + 0.1,
                fillcolor="rgba(0, 0, 255, 0.3)",
                line=dict(color="blue" if i else "red")
            ) for i in range(2)
        ]
    )

    return fig

@app.callback(
    Output('mll-plot', 'figure'),
    Input('scatter-plot', 'relayoutData')
)
def update_mll(relayout_data):
    """ Updates the 3D MLL visualization based on new (x1, y1) and (x2, y2). """

    global initial_points

    x1, x2 = initial_points['x']
    y1, y2 = initial_points['y']    
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
    