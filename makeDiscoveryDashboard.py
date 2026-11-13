import io
import os
import sys
import csv
import base64
import glob
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'geodesicBeamSearch'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'perceptualSimilarity'))

from ODEClass import ODE
from sharedConstants import (invertFlatten, EVAL_REFERENCE_NORM, W_COLUMNS,
                                     NOVELTY_DISTANCE_THRESHOLD_CURVATURE, RELIABLE_MEAN_SELF_DISTANCE_CURVATURE)
from dashboardStyle import (SURFACE, PAGE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BORDER,
                             ACCENT, BASE_FONT, CARD_STYLE, LABEL_STYLE, VALUE_STYLE, DETAIL_STYLE,
                             CHART_CARD_STYLE, EQUATION_BOX_STYLE, PAGE_STYLE, TITLE_STYLE,
                             CONTROL_LABEL_STYLE, TAB_STYLE, TAB_SELECTED_STYLE, statCard, errorFigure)
from renderAndEmbed import renderAllViews, CAMERA_ANGLES

# Ordinal ramp for Round (1 = earliest/lightest -> 5 = latest/darkest), step250-450 from palette.md
ROUND_RAMP = {1: '#86b6ef', 2: '#6da7ec', 3: '#5598e7', 4: '#3987e5', 5: '#2a78d6'}


# More than one search output tree can exist at once (e.g. a renamed-aside historical run next to
# the live 'output' tree) - a dropdown in the app (see
# 'output-dir-dropdown' below) picks between them live, no restart needed. Backups (named with
# '_backup_') are excluded so old renamed-aside runs don't clutter the list.
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


# NearestReference isn't always a NewHit_* row (in df) - 28 of 1080 discoveries' nearest match is
# one of the original catalogued attractors instead (e.g. 'hadley', 'sprottr'), which live in
# attractorDataset.csv, not allUniqueHits.csv. Loaded the same way attractorSimilarity.py's
# loadDatasetAttractors does (fitToCube=False - raw, unscaled coefficients) so the "nearest
# attractor" overlay can resolve either kind of reference. Independent of which output tree is
# selected, so loaded once rather than on every switch.
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


CATALOGED_ATTRACTORS = loadCatalogedAttractors()


# Everything below depends on which output tree is selected - called once at startup and again
# every time 'output-dir-dropdown' changes (see onOutputDirChange), reassigning these as globals
# so the rest of the app (buildOdeForAnyLabel, savedInitialPos, updateDiscovery, ...) keeps reading
# them the same way regardless of whether this is the first load or a live switch.
def loadData(outputDir):
    global OUTPUT_DIR, ALL_HITS_PATH, df, KNOWN_DF_LABELS, HAS_TRAJECTORY_LABELS
    global TOTAL_DISCOVERIES, ROUNDS_COMPLETED, MEAN_LLE, MEAN_CORR_DIM, TOP_ROW
    OUTPUT_DIR = outputDir
    ALL_HITS_PATH = os.path.join(OUTPUT_DIR, 'allUniqueHits.csv')
    df = pd.read_csv(ALL_HITS_PATH)
    df = df.sort_values('LLEMean', ascending=False).reset_index(drop=True)
    KNOWN_DF_LABELS = set(df['Label'])
    TOTAL_DISCOVERIES = len(df)
    ROUNDS_COMPLETED = int(df['Round'].max())
    MEAN_LLE = df['LLEMean'].mean()
    MEAN_CORR_DIM = df['CorrDimMean'].mean()
    TOP_ROW = df.iloc[0]

    # One glob instead of one per row - used by the "has saved trajectory" filter and to warn
    # the Explorer when a selected discovery has no recorded run to show without re-integrating.
    trajPaths = glob.glob(os.path.join(OUTPUT_DIR, 'round_*', 'discoveryTrajectories', '*.npz'))
    HAS_TRAJECTORY_LABELS = set(os.path.splitext(os.path.basename(p))[0] for p in trajPaths)


loadData(_defaultOutputDir(_listOutputDirs(BASE_DIR)))

def axisStyle(title):
    return dict(title=dict(text=title, font=dict(color=INK_SECONDARY)), gridcolor=GRIDLINE,
                zerolinecolor=GRIDLINE, tickfont=dict(color=INK_MUTED), linecolor=GRIDLINE)


# Wider card than statCard: alongside the raw novelty Distance, shows the two reference points it's
# judged against - RELIABLE_MEAN_SELF_DISTANCE_CURVATURE (how far two runs of the SAME attractor
# land from each other - the "noise floor") and NOVELTY_DISTANCE_THRESHOLD_CURVATURE (the cutoff a
# candidate's distance must clear to count as distinct) - plus how far past that cutoff this
# discovery's distance sits. Curvature drives the novelty decision (Distance/NearestReference are
# both curvature-based); D2Distance/D2NearestReference are logged alongside for comparison.
def nearestAttractorCard(row):
    distance = row['Distance']
    pctOverThreshold = ((distance - NOVELTY_DISTANCE_THRESHOLD_CURVATURE)
                         / NOVELTY_DISTANCE_THRESHOLD_CURVATURE * 100)
    return html.Div([
        html.Div('Nearest known attractor', style=LABEL_STYLE),
        html.Div(f"{row['NearestReference']}  (d={distance:.4f})", style=VALUE_STYLE),
        html.Div(f'Self-distance baseline: {RELIABLE_MEAN_SELF_DISTANCE_CURVATURE:.4f}', style=DETAIL_STYLE),
        html.Div(f'Novelty threshold: {NOVELTY_DISTANCE_THRESHOLD_CURVATURE:.4f}', style=DETAIL_STYLE),
        html.Div(f'Exceeds threshold by: {pctOverThreshold:.0f}%', style=DETAIL_STYLE),
    ], style={**CARD_STYLE, 'minWidth': '260px'})


def baseLayout(fig, title, xTitle, yTitle, showlegend=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK_PRIMARY)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=BASE_FONT,
        xaxis=axisStyle(xTitle), yaxis=axisStyle(yTitle),
        margin=dict(l=50, r=20, t=45, b=45), showlegend=showlegend,
        legend=dict(font=dict(color=INK_SECONDARY)),
        bargap=0.15,
    )
    return fig


def makeDiscoveriesPerRoundFig():
    counts = df['Round'].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[f'Round {r}' for r in counts.index], y=counts.values,
                            marker=dict(color=ACCENT)))
    return baseLayout(fig, 'New unique discoveries per round', '', 'Discoveries')


def makeHistogramFig(column, title, xTitle, refLine=None, refLabel=None):
    fig = go.Figure(go.Histogram(x=df[column], marker=dict(color=ACCENT), nbinsx=30))
    if refLine is not None:
        fig.add_vline(x=refLine, line=dict(color=INK_SECONDARY, width=1, dash='dash'),
                      annotation=dict(text=refLabel, font=dict(color=INK_SECONDARY, size=11)))
    return baseLayout(fig, title, xTitle, 'Count')


def makeScatterFig():
    fig = go.Figure()
    for roundNum in sorted(df['Round'].unique()):
        sub = df[df['Round'] == roundNum]
        fig.add_trace(go.Scatter(
            x=sub['LLEMean'], y=sub['CorrDimMean'], mode='markers', name=f'Round {roundNum}',
            marker=dict(size=7, color=ROUND_RAMP.get(int(roundNum), ACCENT),
                        line=dict(width=1, color=SURFACE)),
        ))
    return baseLayout(fig, 'Chaos strength vs. fractal dimension, by round',
                       'Mean Lyapunov exponent (LLE)', 'Mean correlation dimension', showlegend=True)


# overviewLayout/explorerLayout are functions, not fixed values, so onOutputDirChange can rebuild
# them fresh against whichever df loadData() just populated - a plain module-level html.Div would
# only ever reflect whatever was loaded at import time.
def buildOverviewLayout():
    return html.Div([
        html.Div([
            statCard('Unique discoveries', f'{TOTAL_DISCOVERIES:,}'),
            statCard('Rounds completed', f'{ROUNDS_COMPLETED}'),
            statCard('Mean LLE', f'{MEAN_LLE:.4f}'),
            statCard('Most chaotic', f"{TOP_ROW['Label']} ({TOP_ROW['LLEMean']:.3f})"),
            statCard('Mean correlation dim.', f'{MEAN_CORR_DIM:.3f}'),
        ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginBottom': '18px'}),

        html.Div([
            dcc.Graph(figure=makeDiscoveriesPerRoundFig(), style={'height': '340px'}),
        ], style=CHART_CARD_STYLE),

        html.Div([
            html.Div([dcc.Graph(figure=makeHistogramFig('LLEMean', 'Distribution of chaos strength (LLE)', 'LLE'),
                                 style={'height': '320px'})], style=CHART_CARD_STYLE),
            html.Div([dcc.Graph(figure=makeHistogramFig('CorrDimMean', 'Distribution of correlation dimension',
                                                          'Correlation dimension', refLine=1.0,
                                                          refLabel='integer dim. = non-fractal'),
                                 style={'height': '320px'})], style=CHART_CARD_STYLE),
            html.Div([dcc.Graph(figure=makeHistogramFig('Distance', 'Distance from nearest known/prior attractor',
                                                          'Novelty distance'),
                                 style={'height': '320px'})], style=CHART_CARD_STYLE),
        ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'margin': '14px 0'}),

        html.Div([
            dcc.Graph(figure=makeScatterFig(), style={'height': '420px'}),
        ], style=CHART_CARD_STYLE),

        html.Div([
            html.H3('Top 20 most chaotic discoveries', style={'color': INK_PRIMARY, 'fontSize': '16px'}),
            dash_table.DataTable(
                columns=[{'name': c, 'id': c} for c in
                         ['Label', 'Round', 'LLEMean', 'LLEPassRate', 'CorrDimMean', 'NearestReference', 'Distance']],
                data=df.head(20)[['Label', 'Round', 'LLEMean', 'LLEPassRate', 'CorrDimMean', 'NearestReference',
                                   'Distance']].round(4).to_dict('records'),
                style_cell={'backgroundColor': SURFACE, 'color': INK_PRIMARY, 'border': f'1px solid {GRIDLINE}',
                            'fontFamily': 'system-ui, sans-serif', 'fontSize': '13px', 'padding': '6px 10px'},
                style_header={'backgroundColor': PAGE, 'color': INK_SECONDARY, 'fontWeight': '600',
                              'border': f'1px solid {GRIDLINE}'},
            ),
        ], style={**CHART_CARD_STYLE, 'marginTop': '14px'}),
    ], style={'padding': '10px'})


def _dropdownOptionsFor(subDf):
    return [
        {'label': f"{row.Label}  (LLE {row.LLEMean:.3f}, round {int(row.Round)})", 'value': row.Label}
        for row in subDf.itertuples()
    ]


def _numberFilterInput(inputId, value, step):
    return dcc.Input(id=inputId, type='number', value=value, step=step, debounce=True,
                      style={'width': '80px'})


# A plain dropdown over 12,200+ labels is unusable for finding anything specific - these filters
# narrow it to a manageable shortlist before the user even opens it. They only ever touch
# 'discovery-dropdown' options/value (see filterDiscoveries below), never df itself.
def buildFilterPanel():
    maxLLE = round(float(df['LLEMean'].max()), 2)
    maxDist = round(float(df['Distance'].max()), 3)
    return html.Div([
        html.Div([
            html.Label('Label contains', style=CONTROL_LABEL_STYLE),
            dcc.Input(id='filter-label', type='text', placeholder='e.g. NewHit_42', debounce=True,
                       style={'width': '140px'}),
        ]),
        html.Div([
            html.Label('Round', style=CONTROL_LABEL_STYLE),
            html.Div([_numberFilterInput('filter-round-min', 1, 1), html.Span(' to ', style={'color': INK_SECONDARY}),
                      _numberFilterInput('filter-round-max', ROUNDS_COMPLETED, 1)],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}),
        ]),
        html.Div([
            html.Label('LLE (mean)', style=CONTROL_LABEL_STYLE),
            html.Div([_numberFilterInput('filter-lle-min', 0, 0.1), html.Span(' to ', style={'color': INK_SECONDARY}),
                      _numberFilterInput('filter-lle-max', maxLLE, 0.1)],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}),
        ]),
        html.Div([
            html.Label('Novelty distance', style=CONTROL_LABEL_STYLE),
            html.Div([_numberFilterInput('filter-dist-min', 0, 0.01), html.Span(' to ', style={'color': INK_SECONDARY}),
                      _numberFilterInput('filter-dist-max', maxDist, 0.01)],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}),
        ]),
        html.Div([
            html.Label('Nearest reference contains', style=CONTROL_LABEL_STYLE),
            dcc.Input(id='filter-nearest', type='text', placeholder='e.g. bouali', debounce=True,
                       style={'width': '140px'}),
        ]),
        html.Div([
            html.Label('Novelty verdict', style=CONTROL_LABEL_STYLE),
            dcc.RadioItems(
                options=[{'label': ' All', 'value': 'all'}, {'label': ' Methods agree', 'value': 'agree'},
                         {'label': ' Methods disagree', 'value': 'disagree'}],
                value='all', id='filter-disagree', inline=True, style={'color': INK_SECONDARY},
            ),
        ]),
        html.Div([
            html.Label(' ', style=CONTROL_LABEL_STYLE),
            dcc.Checklist(
                options=[{'label': ' Has saved trajectory only', 'value': 'traj'}],
                value=[], id='filter-has-traj', style={'color': INK_SECONDARY},
            ),
        ]),
        html.Div(id='filter-match-count', style={**DETAIL_STYLE, 'alignSelf': 'center', 'whiteSpace': 'nowrap'}),
    ], style={'display': 'flex', 'gap': '22px', 'flexWrap': 'wrap', 'alignItems': 'flex-end',
              'padding': '14px 16px', 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
              'backgroundColor': SURFACE, 'marginBottom': '14px'})


# --- Discovery Explorer: reconstructs the ODE from its stored coefficients (same
# reconstruction/scaling engine.py uses at classification time) and integrates on demand,
# since raw trajectories aren't persisted for most discoveries. ---
def buildExplorerLayout():
    return html.Div([
        buildFilterPanel(),

        html.Div([
            html.Div([
                html.Label('Choose a discovery:', style=CONTROL_LABEL_STYLE),
                html.Div([
                    dcc.Dropdown(options=_dropdownOptionsFor(df), value=TOP_ROW['Label'], id='discovery-dropdown',
                                  style={'flex': '1'}),
                    html.Button(
                        'Random', id='random-button', n_clicks=0,
                        style={
                            'height': '36px', 'padding': '0 14px', 'borderRadius': '6px',
                            'border': f'1px solid {BORDER}', 'backgroundColor': SURFACE, 'color': ACCENT,
                            'fontFamily': 'system-ui, sans-serif', 'fontSize': '13px', 'cursor': 'pointer',
                            'flexShrink': '0',
                        },
                    ),
                ], style={'display': 'flex', 'gap': '8px'}),
            ], style={'width': '480px'}),
            dcc.Checklist(
                options=[{'label': ' Show nearest known/discovered attractor', 'value': 'show'}],
                value=[], id='show-nearest-toggle', style={'color': INK_SECONDARY, 'marginTop': '30px'},
            ),
            dcc.Checklist(
                options=[{'label': ' Use saved trajectory (no re-integration)', 'value': 'use'}],
                value=['use'], id='use-saved-toggle', style={'color': INK_SECONDARY, 'marginTop': '30px'},
            ),
            dcc.RadioItems(
                options=[{'label': ' Same plot', 'value': 'same'},
                         {'label': ' Separate plot (side-by-side)', 'value': 'separate'}],
                value='same', id='nearest-display-mode', inline=True,
                style={'color': INK_SECONDARY, 'marginTop': '30px'},
            ),
            html.Button(
                'Retry integration', id='retry-button', n_clicks=0,
                style={
                    'marginTop': '26px', 'height': '32px', 'padding': '0 14px', 'borderRadius': '6px',
                    'border': f'1px solid {BORDER}', 'backgroundColor': SURFACE, 'color': ACCENT,
                    'fontFamily': 'system-ui, sans-serif', 'fontSize': '13px', 'cursor': 'pointer',
                },
            ),
            html.Button(
                'Show similarity-cascade views', id='cascade-button', n_clicks=0,
                style={
                    'marginTop': '26px', 'height': '32px', 'padding': '0 14px', 'borderRadius': '6px',
                    'border': f'1px solid {BORDER}', 'backgroundColor': SURFACE, 'color': ACCENT,
                    'fontFamily': 'system-ui, sans-serif', 'fontSize': '13px', 'cursor': 'pointer',
                },
            ),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap', 'alignItems': 'flex-start', 'marginBottom': '6px'}),

        html.Div(
            'Note: the search pipeline only writes raw trajectories to disk for the 1,000 most chaotic '
            'discoveries per round. "Use saved trajectory" shows that exact recorded run when one '
            'exists and falls back to a fresh live re-integration when it doesn’t, so the cap '
            'never limits what can be plotted here.',
            style={**DETAIL_STYLE, 'maxWidth': '760px', 'marginBottom': '14px'},
        ),

        html.Div(id='discovery-stats', style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap',
                                               'marginBottom': '14px'}),

        dcc.Loading(children=[
            html.H3('Equation:', style={'color': INK_PRIMARY, 'fontSize': '15px'}),
            html.Div(dcc.Markdown(id='discovery-latex', mathjax=True),
                     style={**EQUATION_BOX_STYLE, 'marginBottom': '28px'}),
        ]),

        dcc.Loading(children=[dcc.Graph(id='discovery-graph', style={'height': '600px'})]),
        # Stacked full-width (not side-by-side) - two 3D plots sharing a narrow flex column left each
        # one too cramped for its own colorbar, which then visibly bled into the other plot's canvas.
        html.Div(dcc.Loading(children=[dcc.Graph(id='discovery-nearest-graph', style={'height': '600px'})]),
                 id='discovery-nearest-container', style={'display': 'none', 'marginTop': '14px'}),

        html.Div(dcc.Loading(children=[html.Div(id='cascade-views-content')]),
                 id='cascade-views-container', style={'marginTop': '14px'}),
    ], style={'padding': '10px'})


def buildBody():
    return dcc.Tabs([
        dcc.Tab(label='Overview & Analysis', children=[buildOverviewLayout()],
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Discovery Explorer', children=[buildExplorerLayout()],
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ])


app = Dash(suppress_callback_exceptions=True)
app.layout = html.Div([
    html.H1('Discovery Dashboard', style=TITLE_STYLE),
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
    Output('discovery-dropdown', 'options'),
    Output('discovery-dropdown', 'value', allow_duplicate=True),
    Output('filter-match-count', 'children'),
    Input('filter-label', 'value'),
    Input('filter-round-min', 'value'),
    Input('filter-round-max', 'value'),
    Input('filter-lle-min', 'value'),
    Input('filter-lle-max', 'value'),
    Input('filter-dist-min', 'value'),
    Input('filter-dist-max', 'value'),
    Input('filter-nearest', 'value'),
    Input('filter-disagree', 'value'),
    Input('filter-has-traj', 'value'),
    State('discovery-dropdown', 'value'),
    prevent_initial_call='initial_duplicate',
)
# Filters never touch df itself, only what the dropdown offers - so every other callback
# (updateDiscovery, showCascadeViews) keeps working against the full dataset regardless of
# how narrow the current filter is. Keeps the current selection if it still matches; otherwise
# jumps to the filtered set's first (most chaotic, since df is sorted by LLEMean) row.
def filterDiscoveries(labelSub, roundMin, roundMax, lleMin, lleMax, distMin, distMax,
                       nearestSub, disagreeMode, hasTrajValue, currentValue):
    sub = df
    if labelSub:
        sub = sub[sub['Label'].str.contains(labelSub, case=False, na=False)]
    if roundMin is not None:
        sub = sub[sub['Round'] >= roundMin]
    if roundMax is not None:
        sub = sub[sub['Round'] <= roundMax]
    if lleMin is not None:
        sub = sub[sub['LLEMean'] >= lleMin]
    if lleMax is not None:
        sub = sub[sub['LLEMean'] <= lleMax]
    if distMin is not None:
        sub = sub[sub['Distance'] >= distMin]
    if distMax is not None:
        sub = sub[sub['Distance'] <= distMax]
    if nearestSub:
        sub = sub[sub['NearestReference'].str.contains(nearestSub, case=False, na=False)]
    if disagreeMode == 'agree':
        sub = sub[~sub['MethodsDisagree']]
    elif disagreeMode == 'disagree':
        sub = sub[sub['MethodsDisagree']]
    if 'traj' in (hasTrajValue or []):
        sub = sub[sub['Label'].isin(HAS_TRAJECTORY_LABELS)]

    matchedLabels = set(sub['Label'])
    newValue = currentValue if currentValue in matchedLabels else (sub.iloc[0]['Label'] if len(sub) else None)
    countText = f'{len(sub):,} of {TOTAL_DISCOVERIES:,} match'
    return _dropdownOptionsFor(sub), newValue, countText


@callback(
    Output('discovery-dropdown', 'value', allow_duplicate=True),
    Input('random-button', 'n_clicks'),
    State('discovery-dropdown', 'options'),
    prevent_initial_call=True,
)
# Picks from whatever the dropdown is currently offering, not the full dataset - so a random
# pick always respects the active filters instead of fighting them.
def pickRandomDiscovery(nClicks, options):
    if not options:
        return dash.no_update
    return random.choice(options)['value']


# Resolves a label to (ode, scaleBy) whether it's a NewHit_* discovery (coefficients in df) or a
# catalogued attractor name (coefficients in CATALOGED_ATTRACTORS) - see the NearestReference note
# above. Both are rescaled to EVAL_REFERENCE_NORM the same way, so a discovery and its nearest
# match sit on a comparable time/space scale when overlaid on the same plot. Returns (None, None)
# if the label matches neither, which callers must handle - not expected to happen given the
# current dataset, but NearestReference is data, not a guarantee.
def buildOdeForAnyLabel(label):
    if label in KNOWN_DF_LABELS:
        flatW = df.loc[df['Label'] == label, W_COLUMNS].iloc[0].to_numpy(dtype=np.float64)
        W = invertFlatten(flatW)
    elif label.lower() in CATALOGED_ATTRACTORS:
        W = CATALOGED_ATTRACTORS[label.lower()]
    else:
        return None, None
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(W))
    return ODE(W=W * scaleBy, doFitToCube=False), scaleBy


def buildOdeForLabel(label):
    row = df[df['Label'] == label].iloc[0]
    ode, scaleBy = buildOdeForAnyLabel(label)
    return ode, scaleBy, row


# A discovery's own saved trajectory (round_*/discoveryTrajectories/{label}.npz - see
# engine.py's saveDiscoveryTrajectoriesBatch) starts from a point already confirmed not to blow up
# under that discovery's own W, so it's a reliable last-resort IC when fresh random draws keep
# failing. Only exists for the pipeline's top MAX_SAVED_DISCOVERY_TRAJECTORIES-per-round discoveries
# (see the storage-cap note in the Explorer layout), so most labels won't have one - that's fine,
# it's a bonus fallback, not something callers can rely on being there.
def savedInitialPos(label):
    matches = glob.glob(os.path.join(OUTPUT_DIR, 'round_*', 'discoveryTrajectories', f'{label}.npz'))
    if not matches:
        return None
    data = np.load(matches[0])
    return [float(data['x'][0]), float(data['y'][0]), float(data['z'][0])]


# The full saved trajectory, not just its starting point - lets the Explorer show the exact
# recorded run (deterministic, matches what the search pipeline actually found) instead of a
# fresh live re-integration. Only exists for the top MAX_SAVED_DISCOVERY_TRAJECTORIES-per-round
# discoveries, same caveat as savedInitialPos.
def loadSavedTrajectory(label):
    matches = glob.glob(os.path.join(OUTPUT_DIR, 'round_*', 'discoveryTrajectories', f'{label}.npz'))
    if not matches:
        return None
    data = np.load(matches[0])
    return data['t'], [data['x'], data['y'], data['z']]


def integrateFrom(ode, scaleBy, initialPos):
    return ode.integrate(initial_pos=initialPos, endTime=1000.0 / scaleBy)


def integrateWithRetries(ode, scaleBy, maxAttempts=6):
    for _ in range(maxAttempts):
        candidate = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        result = integrateFrom(ode, scaleBy, candidate)
        if result is not None:
            return result
    return None


# Tiered IC search for a single trajectory, used once random draws (integrateWithRetries) have
# already failed: fall back to this discovery's own saved-trajectory start point, if one exists.
def integrateWithSavedFallback(ode, scaleBy, label):
    result = integrateWithRetries(ode, scaleBy)
    if result is not None:
        return result
    saved = savedInitialPos(label)
    if saved is not None:
        return integrateFrom(ode, scaleBy, saved)
    return None


# Both traces are time-gradient (never a flat color) so the overlay still reads as two comparable
# "flow through time" shapes rather than one gradient curve plus one flat outline. Plasma for the
# selected trajectory, Oranges for the nearest - a second sequential hue for a second simultaneous
# gradient context, per the dataviz palette (categorical slot 2's hue, one-hue ramp). showscale is
# off by default because two overlapping colorbars for the same "Time" quantity is just clutter -
# separate-plot mode turns it back on since each trace gets its own uncluttered figure there.
def makeTrajectoryTrace(t, y, name, colorscale, showscale):
    lineDict = dict(width=2, color=t, colorscale=colorscale, cmin=t[0], cmax=t[-1], showscale=showscale)
    if showscale:
        tickvals = np.linspace(t[0], t[-1], 6)
        lineDict['colorbar'] = dict(title='Time', tickmode='array', tickvals=tickvals,
                                     tickformat='.0f', ticklabeloverflow='allow')
    return go.Scatter3d(x=y[0], y=y[1], z=y[2], mode='lines', name=name, line=lineDict)


def makeSceneFigure(traces, title, showlegend):
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30), title=title,
        paper_bgcolor=SURFACE, showlegend=showlegend,
        legend=dict(font=dict(color=INK_SECONDARY)),
    )
    return fig


@callback(
    Output('discovery-graph', 'figure'),
    Output('discovery-latex', 'children'),
    Output('discovery-stats', 'children'),
    Output('discovery-nearest-graph', 'figure'),
    Output('discovery-nearest-container', 'style'),
    Input('discovery-dropdown', 'value'),
    Input('show-nearest-toggle', 'value'),
    Input('use-saved-toggle', 'value'),
    Input('nearest-display-mode', 'value'),
    Input('retry-button', 'n_clicks'),
)
# nClicks itself is never read - the button exists purely to re-fire this callback with a fresh
# set of random initial conditions, since integrateWithRetries/integrateWithSavedFallback draw new
# random ICs on every call. Same mechanism doubles as a "try a different random trajectory" button
# even on integrations that already succeeded, not just failed ones. It has no effect while
# "Use saved trajectory" is on, since that path is deterministic (reads the same file every time).
def updateDiscovery(label, showNearestValue, useSavedValue, nearestMode, nClicks):
    showNearest = 'show' in (showNearestValue or [])
    useSaved = 'use' in (useSavedValue or [])
    separate = showNearest and nearestMode == 'separate'
    nearestFig = go.Figure()
    nearestContainerStyle = {'display': 'block' if separate else 'none', 'marginTop': '14px'}

    ode, scaleBy, row = buildOdeForLabel(label)
    latex = f'$${ode.getLatexFormula()}$$'
    stats = [
        statCard('Round', f"{int(row['Round'])}"),
        statCard('LLE (mean)', f"{row['LLEMean']:.4f}"),
        statCard('LLE pass rate', f"{row['LLEPassRate']:.0%}"),
        statCard('Correlation dim.', f"{row['CorrDimMean']:.3f} ± {row['CorrDimStd']:.3f}"),
        nearestAttractorCard(row),
    ]

    result = loadSavedTrajectory(label) if useSaved else None
    if result is not None:
        stats.append(html.Div('Showing the exact saved trajectory for this discovery (no re-integration).',
                               style={**DETAIL_STYLE, 'maxWidth': '260px'}))
    else:
        if useSaved:
            stats.append(html.Div(f'No saved trajectory for {label} - integrated live instead.',
                                   style={**DETAIL_STYLE, 'maxWidth': '260px'}))
        result = integrateWithSavedFallback(ode, scaleBy, label)
    if result is None:
        fig = errorFigure(f'Could not find a stable trajectory for {label} (blew up on every retry)')
        return fig, latex, stats, nearestFig, nearestContainerStyle

    t, y = result
    selectedTrace = makeTrajectoryTrace(t, y, f'Selected: {label}', 'Plasma', showscale=True)
    traces = [selectedTrace]

    nearestLabel = row['NearestReference']
    if showNearest:
        nearestResult = loadSavedTrajectory(nearestLabel) if useSaved else None
        if nearestResult is None:
            nearestOde, nearestScaleBy = buildOdeForAnyLabel(nearestLabel)
            if nearestOde is not None:
                # Tier 1: try the exact point the selected trajectory already starts from - free to
                # attempt, and since both systems were seeded from nearby coefficients (that's why
                # they're "nearest" in the first place) it often lands in a comparable basin, which
                # also makes the overlay a fairer shape comparison than two unrelated starting points.
                sharedInitial = [float(y[0][0]), float(y[1][0]), float(y[2][0])]
                nearestResult = integrateFrom(nearestOde, nearestScaleBy, sharedInitial)
                # Tier 2: independent random search, same as the selected trajectory gets.
                # Tier 3: the nearest discovery's own saved-trajectory start point, if one exists.
                if nearestResult is None:
                    nearestResult = integrateWithSavedFallback(nearestOde, nearestScaleBy, nearestLabel)
        if nearestResult is not None:
            nt, ny = nearestResult
            nearestTrace = makeTrajectoryTrace(nt, ny, f'Nearest: {nearestLabel}', 'Oranges', showscale=separate)
            if separate:
                nearestFig = makeSceneFigure([nearestTrace], f'Nearest: {nearestLabel}', showlegend=False)
            else:
                traces.append(nearestTrace)
        else:
            stats.append(html.Div(
                f'Could not render a trajectory for the nearest attractor ({nearestLabel}) - it '
                f'blew up on every retry.', style={**DETAIL_STYLE, 'maxWidth': '260px'},
            ))

    fig = makeSceneFigure(traces, label, showlegend=len(traces) > 1)
    return fig, latex, stats, nearestFig, nearestContainerStyle


def _imageToDataUri(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


_CASCADE_THUMB_STYLE = {
    'width': '130px', 'height': '130px', 'display': 'block',
    'borderRadius': '6px', 'border': f'1px solid {BORDER}', 'backgroundColor': '#fff',
}


def _cascadeThumb(img):
    return html.Img(src=_imageToDataUri(img), style=_CASCADE_THUMB_STYLE)


# One row of the six fixed camera-angle silhouettes for a single trajectory - the exact images
# the Zernike-moment shortlist and ICP alignment in perceptualSimilarity/noveltyCascade.py
# compare, not the raw curve. Used in "same plot" mode, where the two trajectories are already
# overlaid rather than compared angle-by-angle.
def _cascadeImageRow(rowLabel, images):
    return html.Div([
        html.Div(rowLabel, style={**LABEL_STYLE, 'marginBottom': '8px'}),
        html.Div([
            html.Div([
                _cascadeThumb(img),
                html.Div(f'elev {elev}°, az {azim}°', style={**DETAIL_STYLE, 'textAlign': 'center', 'marginTop': '4px'}),
            ]) for img, (elev, azim) in zip(images, CAMERA_ANGLES)
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}),
    ], style={**CHART_CARD_STYLE, 'marginBottom': '14px'})


# Side-by-side mode: one group per camera angle, discovery and nearest thumbnails placed next to
# each other - so every angle shows both trajectories together instead of two separate blocks of
# six, which is what "separate plot" means for the main 3D graphs too.
def _cascadePairedGrid(label, images, nearestLabel, nearestImages):
    groups = []
    for i, (elev, azim) in enumerate(CAMERA_ANGLES):
        pair = [html.Div([_cascadeThumb(images[i]), html.Div(label, style={**DETAIL_STYLE, 'textAlign': 'center', 'marginTop': '4px'})])]
        if nearestImages is not None:
            pair.append(html.Div([_cascadeThumb(nearestImages[i]),
                                   html.Div(nearestLabel, style={**DETAIL_STYLE, 'textAlign': 'center', 'marginTop': '4px'})]))
        groups.append(html.Div([
            html.Div(f'elev {elev}°, az {azim}°', style={**DETAIL_STYLE, 'textAlign': 'center',
                                                           'marginBottom': '6px', 'fontWeight': '600'}),
            html.Div(pair, style={'display': 'flex', 'gap': '8px'}),
        ], style={'padding': '10px', 'border': f'1px solid {BORDER}', 'borderRadius': '8px'}))
    return html.Div(groups, style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '14px'})


def buildCascadeViewPanel(label, y, nearestLabel, ny, sideBySide):
    images = renderAllViews(y)
    nearestImages = renderAllViews(ny) if ny is not None else None

    if sideBySide:
        body = [_cascadePairedGrid(label, images, nearestLabel, nearestImages)]
    else:
        body = [_cascadeImageRow(f'Discovery: {label}', images)]
        if nearestImages is not None:
            body.append(_cascadeImageRow(f'Nearest: {nearestLabel}', nearestImages))

    if ny is None and nearestLabel is not None:
        body.append(html.Div(f'Could not render cascade views for the nearest attractor ({nearestLabel}) - '
                              f'it blew up on every retry.', style={**DETAIL_STYLE, 'maxWidth': '420px'}))

    return html.Div([
        html.H3('What the similarity cascade sees', style={'color': INK_PRIMARY, 'fontSize': '15px'}),
        html.Div(
            'The novelty cascade never looks at the raw 3D curve - it renders each trajectory from '
            'these six fixed camera angles as a black silhouette, summarizes each with Zernike '
            'moments for a fast shortlist, then runs ICP alignment on the shortlist survivors to '
            'get the final match. These are exactly those renders.',
            style={**DETAIL_STYLE, 'maxWidth': '760px', 'marginBottom': '12px'},
        ),
        html.Div(body),
    ])


@callback(
    Output('cascade-views-content', 'children'),
    Input('cascade-button', 'n_clicks'),
    State('discovery-dropdown', 'value'),
    State('show-nearest-toggle', 'value'),
    State('use-saved-toggle', 'value'),
    State('nearest-display-mode', 'value'),
    prevent_initial_call=True,
)
# Deliberately independent of updateDiscovery - re-fetches its own trajectories rather than
# sharing that callback's, so switching discoveries doesn't pay this render cost until the
# button is actually clicked again.
def showCascadeViews(nClicks, label, showNearestValue, useSavedValue, nearestMode):
    showNearest = 'show' in (showNearestValue or [])
    useSaved = 'use' in (useSavedValue or [])

    ode, scaleBy, row = buildOdeForLabel(label)
    result = loadSavedTrajectory(label) if useSaved else None
    if result is None:
        result = integrateWithSavedFallback(ode, scaleBy, label)
    if result is None:
        return html.Div(f'Could not render cascade views for {label} (no stable trajectory found).',
                         style={**DETAIL_STYLE, 'maxWidth': '420px'})
    _, y = result

    nearestLabel = row['NearestReference'] if showNearest else None
    ny = None
    if showNearest:
        nearestResult = loadSavedTrajectory(nearestLabel) if useSaved else None
        if nearestResult is None:
            nearestOde, nearestScaleBy = buildOdeForAnyLabel(nearestLabel)
            if nearestOde is not None:
                nearestResult = integrateWithSavedFallback(nearestOde, nearestScaleBy, nearestLabel)
        if nearestResult is not None:
            _, ny = nearestResult

    return buildCascadeViewPanel(label, y, nearestLabel, ny, sideBySide=(nearestMode == 'separate'))


if __name__ == '__main__':
    app.run(debug=False)
