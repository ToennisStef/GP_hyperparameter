import json
from textwrap import dedent as d
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.gp_hyperparameter.updates import update_global_points, update_GP_plot
from src.gp_hyperparameter.funcs import calc_GP
import numpy as np

app = Dash(__name__)

# Initial values for points
N_points = 2

class DataPoints:
    def __init__(self, 
                 x=None, 
                 y=None
                 ):
        if x is None:
            self.x = np.random.uniform(-2.5, 2.5)
        else:
            self.x = x
        if y is None:
            self.y = np.random.uniform(-2.5, 2.5)
        else:
            self.y = y

DataPoints = [DataPoints() for _ in range(N_points)]
        
x = [point.x for point in DataPoints]

styles = {'pre': {'border': 'thin lightgrey solid', 'overflowX': 'scroll'}}

app.layout = html.Div([
    # Header
    html.H1(children='MLL Visualization', 
            style={'textAlign': 'center'}),
    # Description
    html.Div(children='Visualization of the log marginal likelihood of a Gaussian Process with respect to the hyperparameters. The plot shows the log marginal likelihood as a function of the hyperparameters σₙ and θₗ. The red line shows the maximum log marginal likelihood for fixed θₗ and the blue line shows the maximum log marginal likelihood for fixed σₙ. The plot updates as the sliders are adjusted. You can adjust the position of the observation and the observation value',
             style={'textAlign': 'center'}),
    # Scatter plot
    dcc.Graph(id='scatter-plot', 
              config={'editable': True, 'edits': {'shapePosition': True}},
              style={'width': '45vw', 'aspect-ratio': '1/1'}),
    # Noise inference dropdown
    dcc.Dropdown(id='noise_inference', 
                 options=[
                     {'label': 'Fixed Zero Noise', 'value': 'fixed zero'},
                     {'label': 'Inferred Noise', 'value': 'inferred'}
                 ],
                 value='inferred'),
    # relayoutData
    html.Div(
        className='six columns',
        children=[
            html.Div(
                [
                    dcc.Markdown(
                        d("""
                **Zoom and Relayout Data**

            """)),
                    html.Pre(id='relayout-data', style=styles['pre']),
                ]
            )
        ]
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'relayoutData'),
    Input('noise_inference', 'value')
)
def update_scatter(relayoutData, noise_inference):
    """ Updates the scatter plot and captures new (x1, y1) and (x2, y2) positions. """
    
    
    update_global_points(relayoutData, DataPoints)
    
    x = [point.x for point in DataPoints]
    y = [point.y for point in DataPoints]
    
    
    X_plot, Y_mean, Y_var = calc_GP(noise_inference, x, y)
    
        
    # Create scatter plot with draggable points
    fig = go.Figure()

    update_GP_plot(fig, x, y, X_plot, Y_mean, Y_var)
    
    # Set axis range
    fig.update_xaxes(range=[-5, 5])
    fig.update_yaxes(range=[-5, 5])
    
    # Update legend position        
    fig.update_legends(
        x = 0.01,
        y = 0.99,
        traceorder='normal')
    
    return fig

@app.callback(
    Output('relayout-data', 'children'),
    [Input('scatter-plot', 'relayoutData')])
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)