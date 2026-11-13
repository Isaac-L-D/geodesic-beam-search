import csv
from ODEClass import ODE
from dash import Dash, html, dcc, callback, Output, Input
from dashboardStyle import (INK_PRIMARY, EQUATION_BOX_STYLE, PAGE_STYLE, TITLE_STYLE,
                             CONTROL_LABEL_STYLE, errorFigure, styleFigureChrome)

dataset = {}
with open('attractorDataset.csv', 'r') as file:
    reader = csv.reader(file)
    fields = next(reader)
    for row in reader:
        ode, name = ODE.loadFromCSVString(row)
        dataset[name] = ode
keys = list(dataset.keys())

app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1('View of Dataset of Attractors', style=TITLE_STYLE),

    html.Div([
        html.Div([
            html.Label('Choose an attractor:', style=CONTROL_LABEL_STYLE),
            dcc.Dropdown(options=keys, value=keys[0], id='dropdown-selection'),
        ], style={'width': '480px'}),
        html.Div([
            html.Label('Plot one or multiple trajectories?', style=CONTROL_LABEL_STYLE),
            dcc.RadioItems(options=['One', 'Multiple'], value='One', id='radioSelection', inline=True),
        ], style={'marginTop': '2px'}),
    ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap', 'alignItems': 'flex-start', 'marginBottom': '14px'}),

    dcc.Loading(children=[
        html.H3('Equation:', style={'color': INK_PRIMARY, 'fontSize': '15px'}),
        html.Div(dcc.Markdown(id='latex', mathjax=True), style={**EQUATION_BOX_STYLE, 'marginBottom': '28px'}),
    ]),

    dcc.Loading(children=[dcc.Graph(id='graph-content', style={'height': '650px'})]),
], style=PAGE_STYLE)


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value'),
    Input('radioSelection', 'value'),
)
def update_graph_dropdown(graphToGrab, singleOrMulti):
    ode: ODE = dataset[graphToGrab]
    try:
        fig = ode.plotMultipleTrajectories(doSave=False) if singleOrMulti == 'Multiple' else ode.plotInteractive(doSave=False)
        if fig is None:
            raise Exception()
        return styleFigureChrome(fig)
    except Exception:
        return errorFigure(f'There was an error loading the figure of the ODE for {graphToGrab}')


@callback(
    Output('latex', 'children'),
    Input('dropdown-selection', 'value'),
)
def updateLatex(graphToGrab):
    ode: ODE = dataset[graphToGrab]
    return f'$${ode.getLatexFormula()}$$'


if __name__ == '__main__':
    app.run(debug=False)
