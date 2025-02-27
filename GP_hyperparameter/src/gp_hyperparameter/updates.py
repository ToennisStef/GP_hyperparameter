import re
import plotly.graph_objects as go
import numpy as np

def update_global_points(relayoutData, DataPoints):
    
    if relayoutData is not None and any("shapes" in key for key in relayoutData):
        S_x_pattern = []
        S_y_pattern = []
        S_x_keys = {}
        S_y_keys = {}
        new_x = {}
        new_y = {}
        for i in range(len(DataPoints)):        
            S_x_pattern.append(re.compile(fr"shapes\[{i}\]\..*x"))  # Matches anything after "shapes[0].x"
            S_y_pattern.append(re.compile(rf"shapes\[{i}\]\..*y"))  # Matches anything after "shapes[0].y"

            for key in relayoutData.keys():
                if S_x_pattern[i].match(key):
                    if i not in S_x_keys:
                        S_x_keys[i] = []
                    S_x_keys[i].append(key)
                if S_y_pattern[i].match(key):
                    if i not in S_y_keys:
                        S_y_keys[i] = []
                    S_y_keys[i].append(key)
        
        for key in S_x_keys.keys():
            new_x[key] = (relayoutData[S_x_keys[key][0]]+relayoutData[S_x_keys[key][1]])/2
            
        for key in S_y_keys.keys():
            new_y[key] = (relayoutData[S_y_keys[key][0]]+relayoutData[S_y_keys[key][1]])/2
            
        
        for key in new_x.keys():
            DataPoints[key].x = new_x[key]  # Modify the global variable
            
        for key in new_y.keys():
            DataPoints[key].y = new_y[key]    # Modify the global variable
            
def update_GP_plot(fig, x, y, X_plot, Y_mean, Y_var):
    # Create scatter plot with draggable points
    
    # Add the nice looking scatter plots
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=10, color='black'), 
        name="Draggable Points"
    ))

    # Add mean prediction line
    fig.add_trace(go.Scatter(
        x=X_plot,
        y=Y_mean,
        mode='lines',
        name='Mean Prediction',
        line=dict(color='black')
    ))

    # Add variance prediction as a filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([X_plot, X_plot[::-1]]),
        y=np.concatenate([Y_mean - 1.96 * np.sqrt(Y_var), 
                          (Y_mean + 1.96 * np.sqrt(Y_var))[::-1]]),
        fill='toself',
        fillcolor='rgba(100, 100, 100, 0.2)',
        line=dict(color='rgba(100, 100, 100,0.2)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))

    # Add draggable shape/ Adds two circles via dictioanry comprehension
    fig.update_layout(
        shapes=[
            dict(
                type="circle",
                xref="x", yref="y",
                x0=x[i]-0.1, x1=x[i]+0.1,
                y0=y[i]-0.1, y1=y[i]+0.1,
                # fillcolor="rgba(0, 0, 0, 0)", # Commented out for now
                line=dict(color="blue")
            ) for i in range(len(x))
        ]
    )