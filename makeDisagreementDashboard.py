import os
import sys
import csv
import glob
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, callback, Output, Input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'geodesicBeamSearch'))

from ODEClass import ODE
from sharedConstants import (invertFlatten, EVAL_REFERENCE_NORM, W_COLUMNS,
                              NOVELTY_DISTANCE_THRESHOLD, NOVELTY_DISTANCE_THRESHOLD_CURVATURE)
from dashboardStyle import (SURFACE, PAGE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BORDER,
                             ACCENT, BASE_FONT, CARD_STYLE, LABEL_STYLE, VALUE_STYLE, DETAIL_STYLE,
                             CHART_CARD_STYLE, PAGE_STYLE, TITLE_STYLE, CONTROL_LABEL_STYLE,
                             statCard, errorFigure)

# More than one search output tree can exist at once (e.g. a renamed-aside historical run next to
# the live 'output' tree) - a dropdown in the app (see 'output-dir-dropdown' below) picks between
# them live, no restart needed. Backups (named with '_backup_') are excluded so old renamed-aside
# runs don't clutter the list.
BASE_DIR = os.path.join(SCRIPT_DIR, 'geodesicBeamSearch')


# Matched by presence of latticeSamplerResults.txt (a marker every variant writes, regardless of
# naming convention) rather than hardcoding a single directory name.
def _listOutputDirs(baseDir):
    return sorted(d for d in glob.glob(os.path.join(baseDir, '*'))
                  if os.path.isdir(d) and '_backup_' not in os.path.basename(d)
                  and os.path.exists(os.path.join(d, 'latticeSamplerResults.txt')))


# Defaults to whichever tree's log was written to most recently - the run someone actively started
# is far more likely to be what they want to look at first than an older, finished one.
def _defaultOutputDir(candidates):
    if not candidates:
        return os.path.join(BASE_DIR, 'output')

    def freshness(d):
        logPath = os.path.join(d, 'latticeSamplerResults.txt')
        return os.path.getmtime(logPath) if os.path.exists(logPath) else 0
    return max(candidates, key=freshness)


ATTRACTOR_DATASET_PATH = os.path.join(SCRIPT_DIR, 'attractorDataset.csv')


def loadCatalogedAttractors(path=ATTRACTOR_DATASET_PATH):
    catalog = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ode, name = ODE.loadFromCSVString(row, fitToCube=False)
            catalog[name] = ode.W
    return catalog


# Independent of which output tree is selected, so loaded once rather than on every switch.
CATALOGED_ATTRACTORS = loadCatalogedAttractors()


# Everything below depends on which output tree is selected - called once at startup and again
# every time 'output-dir-dropdown' changes (see onOutputDirChange), reassigning these as globals so
# the rest of the app (buildOdeForAnyLabel, updateDisagreement, ...) keeps reading them the same
# way regardless of whether this is the first load or a live switch.
#
# Views only, no approve/reject workflow - runRound already made the actual novelty call before
# any of this is written, and disagreements.csv is a record of what it decided and why, not a
# queue of decisions waiting to be made.
def loadData(outputDir):
    global OUTPUT_DIR, ALL_HITS_PATH, df, TOTAL_DISAGREEMENTS, _allHitsDf, KNOWN_HIT_LABELS
    OUTPUT_DIR = outputDir
    ALL_HITS_PATH = os.path.join(OUTPUT_DIR, 'allUniqueHits.csv')

    disagreementPaths = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'round_*', 'disagreements.csv')))
    frames = []
    for path in disagreementPaths:
        roundNum = int(path.split('round_')[1].split(os.sep)[0])
        frame = pd.read_csv(path)
        frame.insert(0, 'Round', roundNum)
        frame.insert(1, 'RowInRound', range(len(frame)))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['Round', 'RowInRound', 'ParentLabel', 'CurvatureVerdict', 'D2Verdict',
                 'CurvatureNearestReference', 'CurvatureDistance', 'D2NearestReference', 'D2Distance',
                 'MeanCurvature', 'LLEMean'] + W_COLUMNS)
    TOTAL_DISAGREEMENTS = len(df)

    _allHitsDf = pd.read_csv(ALL_HITS_PATH) if os.path.exists(ALL_HITS_PATH) else None
    KNOWN_HIT_LABELS = set(_allHitsDf['Label']) if _allHitsDf is not None else set()


loadData(_defaultOutputDir(_listOutputDirs(BASE_DIR)))


# Resolves a reference label (either a NewHit_* discovery or a catalogued attractor name) to an
# ODE, the same way makeDiscoveryDashboard.py's buildOdeForAnyLabel does - see that file for why
# both kinds of reference need handling here.
def buildOdeForAnyLabel(label):
    if label in KNOWN_HIT_LABELS:
        flatW = _allHitsDf.loc[_allHitsDf['Label'] == label, W_COLUMNS].iloc[0].to_numpy(dtype=np.float64)
        W = invertFlatten(flatW)
    elif label.lower() in CATALOGED_ATTRACTORS:
        W = CATALOGED_ATTRACTORS[label.lower()]
    else:
        return None, None
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(W))
    return ODE(W=W * scaleBy, doFitToCube=False), scaleBy


def buildOdeForRow(row):
    flatW = row[W_COLUMNS].to_numpy(dtype=np.float64)
    W = invertFlatten(flatW)
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(W))
    return ODE(W=W * scaleBy, doFitToCube=False), scaleBy


def integrateFrom(ode, scaleBy, initialPos):
    return ode.integrate(initial_pos=initialPos, endTime=1000.0 / scaleBy)


def integrateWithRetries(ode, scaleBy, maxAttempts=6):
    for _ in range(maxAttempts):
        candidate = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        result = integrateFrom(ode, scaleBy, candidate)
        if result is not None:
            return result
    return None


def makeTrajectoryTrace(t, y, name, colorscale, showscale):
    lineDict = dict(width=2, color=t, colorscale=colorscale, cmin=t[0], cmax=t[-1], showscale=showscale)
    if showscale:
        tickvals = np.linspace(t[0], t[-1], 6)
        lineDict['colorbar'] = dict(title='Time', tickmode='array', tickvals=tickvals,
                                     tickformat='.0f', ticklabeloverflow='allow')
    return go.Scatter3d(x=y[0], y=y[1], z=y[2], mode='lines', name=name, line=lineDict)


def makeSceneFigure(traces, title):
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30), title=title,
        paper_bgcolor=SURFACE, showlegend=False,
    )
    return fig


def axisStyle(title):
    return dict(title=dict(text=title, font=dict(color=INK_SECONDARY)), gridcolor=GRIDLINE,
                zerolinecolor=GRIDLINE, tickfont=dict(color=INK_MUTED), linecolor=GRIDLINE)


def makeOverviewFig():
    fig = go.Figure()
    for verdict, color in [('distinct', ACCENT), ('same', INK_MUTED)]:
        sub = df[df['CurvatureVerdict'] == verdict]
        fig.add_trace(go.Scatter(
            x=sub['D2Distance'], y=sub['CurvatureDistance'], mode='markers',
            name=f'Curvature: {verdict}',
            marker=dict(size=7, color=color, line=dict(width=1, color=SURFACE)),
        ))
    fig.add_vline(x=NOVELTY_DISTANCE_THRESHOLD, line=dict(color=INK_SECONDARY, width=1, dash='dash'),
                  annotation=dict(text='D2 threshold', font=dict(color=INK_SECONDARY, size=11)))
    fig.add_hline(y=NOVELTY_DISTANCE_THRESHOLD_CURVATURE, line=dict(color=INK_SECONDARY, width=1, dash='dash'),
                  annotation=dict(text='Curvature threshold', font=dict(color=INK_SECONDARY, size=11)))
    fig.update_layout(
        title=dict(text='Every disagreement: D2 distance vs. curvature distance', font=dict(size=15, color=INK_PRIMARY)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=BASE_FONT,
        xaxis=axisStyle('D2 distance to nearest reference'),
        yaxis=axisStyle('Curvature distance to nearest reference'),
        margin=dict(l=50, r=20, t=45, b=45), showlegend=True,
        legend=dict(font=dict(color=INK_SECONDARY)),
    )
    return fig


# overviewLayout/explorerLayout are functions, not fixed values, so onOutputDirChange can rebuild
# them fresh against whichever df loadData() just populated - a plain module-level html.Div would
# only ever reflect whatever was loaded at import time.
def buildOverviewLayout():
    return html.Div([
        html.Div([
            statCard('Total disagreements', f'{TOTAL_DISAGREEMENTS:,}'),
            statCard('Curvature says distinct, D2 says same',
                     f"{len(df[(df['CurvatureVerdict']=='distinct') & (df['D2Verdict']=='same')]):,}"),
            statCard('D2 says distinct, curvature says same',
                     f"{len(df[(df['CurvatureVerdict']=='same') & (df['D2Verdict']=='distinct')]):,}"),
        ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginBottom': '18px'}),
        html.Div([dcc.Graph(figure=makeOverviewFig(), style={'height': '480px'})], style=CHART_CARD_STYLE),
    ], style={'padding': '10px'})


def buildExplorerLayout():
    dropdownOptions = [
        {'label': f"Round {row.Round} #{row.RowInRound} - parent {row.ParentLabel}  "
                  f"(curvature: {row.CurvatureVerdict}, D2: {row.D2Verdict})",
         'value': f'{row.Round}:{row.RowInRound}'}
        for row in df.itertuples()
    ]
    return html.Div([
        html.Div([
            html.Label('Choose a disagreement:', style=CONTROL_LABEL_STYLE),
            dcc.Dropdown(options=dropdownOptions,
                         value=dropdownOptions[0]['value'] if dropdownOptions else None,
                         id='disagreement-dropdown'),
        ], style={'width': '640px', 'marginBottom': '14px'}),

        html.Div(id='disagreement-stats', style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap',
                                                  'marginBottom': '18px'}),

        html.H3('Disputed candidate', style={'color': INK_PRIMARY, 'fontSize': '15px'}),
        dcc.Loading(children=[dcc.Graph(id='candidate-graph', style={'height': '520px'})]),

        html.Div([
            html.Div([
                html.H3("Curvature's nearest reference", style={'color': INK_PRIMARY, 'fontSize': '15px'}),
                dcc.Loading(children=[dcc.Graph(id='curvature-nearest-graph', style={'height': '480px'})]),
            ], style={'flex': '1', 'minWidth': '420px'}),
            html.Div([
                html.H3("D2's nearest reference", style={'color': INK_PRIMARY, 'fontSize': '15px'}),
                dcc.Loading(children=[dcc.Graph(id='d2-nearest-graph', style={'height': '480px'})]),
            ], style={'flex': '1', 'minWidth': '420px'}),
        ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginTop': '14px'}),
    ], style={'padding': '10px'})


def buildBody():
    return dcc.Tabs([
        dcc.Tab(label='Overview', children=[buildOverviewLayout()], style={'color': INK_SECONDARY},
                selected_style={'color': ACCENT, 'fontWeight': '600'}),
        dcc.Tab(label='Explorer', children=[buildExplorerLayout()], style={'color': INK_SECONDARY},
                selected_style={'color': ACCENT, 'fontWeight': '600'}),
    ])


app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1('Disagreement Viewer', style=TITLE_STYLE),
    html.Div(
        'Every hit where curvature and D2 reached a different novelty verdict, whether or not '
        'curvature ended up accepting it. Curvature drives the actual decision - D2 is shown for '
        'comparison, not as a second vote.',
        style={**DETAIL_STYLE, 'textAlign': 'center', 'maxWidth': '760px', 'margin': '0 auto 20px'},
    ),
    html.Div([
        html.Label('Search output:', style=CONTROL_LABEL_STYLE),
        dcc.Dropdown(
            options=[{'label': os.path.basename(d), 'value': d}
                     for d in _listOutputDirs(BASE_DIR)] or
                    [{'label': os.path.basename(OUTPUT_DIR), 'value': OUTPUT_DIR}],
            value=OUTPUT_DIR, id='output-dir-dropdown', clearable=False,
        ),
    ], style={'width': '420px', 'margin': '0 auto 18px'}),
    html.Div(id='app-body', children=buildBody()),
], style=PAGE_STYLE)


@callback(Output('app-body', 'children'), Input('output-dir-dropdown', 'value'))
def onOutputDirChange(chosenDir):
    loadData(chosenDir)
    return buildBody()


@callback(
    Output('candidate-graph', 'figure'),
    Output('disagreement-stats', 'children'),
    Output('curvature-nearest-graph', 'figure'),
    Output('d2-nearest-graph', 'figure'),
    Input('disagreement-dropdown', 'value'),
)
def updateDisagreement(value):
    if value is None:
        empty = errorFigure('No disagreements logged yet.')
        return empty, [], empty, empty

    roundNum, rowInRound = value.split(':')
    row = df[(df['Round'] == int(roundNum)) & (df['RowInRound'] == int(rowInRound))].iloc[0]

    stats = [
        statCard('Round', f"{int(row['Round'])}"),
        statCard('Curvature verdict', row['CurvatureVerdict']),
        statCard('D2 verdict', row['D2Verdict']),
        statCard('Curvature distance', f"{row['CurvatureDistance']:.4f}  (threshold {NOVELTY_DISTANCE_THRESHOLD_CURVATURE:.4f})"),
        statCard('D2 distance', f"{row['D2Distance']:.4f}  (threshold {NOVELTY_DISTANCE_THRESHOLD:.4f})"),
        statCard('Mean curvature', f"{row['MeanCurvature']:.4f}" if pd.notna(row['MeanCurvature']) else 'n/a'),
        statCard('LLE (mean)', f"{row['LLEMean']:.4f}"),
    ]

    candidateOde, candidateScaleBy = buildOdeForRow(row)
    candidateResult = integrateWithRetries(candidateOde, candidateScaleBy)
    if candidateResult is None:
        candidateFig = errorFigure('Could not find a stable trajectory for this candidate.')
    else:
        t, y = candidateResult
        candidateFig = makeSceneFigure([makeTrajectoryTrace(t, y, 'Candidate', 'Plasma', showscale=True)],
                                        f"Round {row['Round']} candidate (parent {row['ParentLabel']})")

    def nearestFig(label, colorscale):
        ode, scaleBy = buildOdeForAnyLabel(label)
        if ode is None:
            return errorFigure(f'Could not resolve reference "{label}".')
        result = integrateWithRetries(ode, scaleBy)
        if result is None:
            return errorFigure(f'Could not find a stable trajectory for "{label}".')
        t, y = result
        return makeSceneFigure([makeTrajectoryTrace(t, y, label, colorscale, showscale=True)], label)

    curvFig = nearestFig(row['CurvatureNearestReference'], 'Plasma')
    d2Fig = nearestFig(row['D2NearestReference'], 'Oranges')

    return candidateFig, stats, curvFig, d2Fig


if __name__ == '__main__':
    app.run(debug=False)
