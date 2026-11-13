from dash import html
import plotly.graph_objects as go

# Shared visual language for this project's Dash dashboards (see the dataviz skill: light chart
# palette, single accent hue). Both makeDiscoveryDashboard.py and makeDatasetDashboard.py import
# these rather than redefining their own copies, so the two apps can't visually drift apart.
SURFACE = '#fcfcfb'
PAGE = '#f9f9f7'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BORDER = 'rgba(11,11,11,0.10)'
ACCENT = '#2a78d6'

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
BASE_FONT = dict(family=FONT_FAMILY, color=INK_PRIMARY)

CARD_STYLE = {
    'backgroundColor': SURFACE, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
    'padding': '16px 20px', 'flex': '1', 'minWidth': '160px',
}
LABEL_STYLE = {'color': INK_SECONDARY, 'fontSize': '13px', 'marginBottom': '6px'}
VALUE_STYLE = {'color': INK_PRIMARY, 'fontSize': '26px', 'fontWeight': '600'}
DETAIL_STYLE = {'color': INK_SECONDARY, 'fontSize': '12px', 'marginTop': '2px'}
CHART_CARD_STYLE = {
    'backgroundColor': SURFACE, 'border': f'1px solid {BORDER}', 'borderRadius': '8px',
    'padding': '12px', 'flex': '1', 'minWidth': '380px',
}
# The bordered, scrollable box used for equation display - shared because both dashboards render
# arbitrarily long polynomials (up to 27 monomials per equation) that don't fit on one line.
EQUATION_BOX_STYLE = {
    'maxHeight': '220px', 'overflow': 'auto', 'backgroundColor': SURFACE,
    'border': f'1px solid {BORDER}', 'borderRadius': '8px', 'padding': '12px 16px',
}

PAGE_STYLE = {'backgroundColor': PAGE, 'minHeight': '100vh', 'fontFamily': FONT_FAMILY, 'padding': '18px'}
TITLE_STYLE = {'textAlign': 'center', 'color': INK_PRIMARY}
CONTROL_LABEL_STYLE = {'color': INK_SECONDARY}
TAB_STYLE = {'color': INK_SECONDARY}
TAB_SELECTED_STYLE = {'color': ACCENT, 'fontWeight': '600'}


def statCard(label, value):
    return html.Div([html.Div(label, style=LABEL_STYLE), html.Div(value, style=VALUE_STYLE)], style=CARD_STYLE)


# Applied to any go.Figure returned by ODEClass's own plotting methods (which build traces with
# their own coloring, e.g. Plasma time-gradients) so the surrounding chrome - background, error
# text - matches the dashboard shell without touching ODEClass.py's plotting logic itself.
def styleFigureChrome(fig):
    fig.update_layout(paper_bgcolor=SURFACE, font=BASE_FONT)
    return fig


def errorFigure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        font=dict(family=FONT_FAMILY, size=16, color=INK_SECONDARY), align='center',
    )
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
    return fig
