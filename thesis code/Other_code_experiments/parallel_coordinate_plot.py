import os
import sys
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import plotly.graph_objects as go

init_notebook_mode(connected=True)

CURRENT_DIR = os.getcwd()
df = pd.read_csv(f'{CURRENT_DIR}/results.csv')

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['accuracy'],
                  colorbar = [],
                   colorscale = [[0, '#6C9E12'], ##
                                [0.25,'#0D5F67'], ##
                                [0.5,'#AA1B13'], ##
                                [0.75, '#69178C'], ##
                                [1, '#DE9733']]),
        dimensions = list([
            dict(range = [0,12],
                label = 'Conv_layers', values = df['conv_layers']),
            dict(range = [8,64],
                label = 'filter_number', values = df['filters']),
            dict(range = [0.2,0.8],
                label = 'dropout_rate', values = df['dropout']),
             dict(range = [0.1,1.0],
                label = 'accuracy', values = df['accuracy'])
        ])
    )
)


fig.update_layout(
    plot_bgcolor = '#E5E5E5',
    paper_bgcolor = '#E5E5E5',
    title="Parallel Coorinates Plot"
)

# print the plot
fig.show()
