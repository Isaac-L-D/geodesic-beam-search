import os
import sys
import glob
import numpy as np
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
DEFAULT_PLOT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'Plots', 'discoveries')


# More than one search output tree can exist at once (e.g. a renamed-aside historical run next to
# the live 'output' tree) - ask which to view rather than silently hardcoding one. Matched by
# presence of latticeSamplerResults.txt (a marker every variant writes) rather than hardcoding a
# single directory name. Backups (named with '_backup_') are excluded so old renamed-aside runs
# don't clutter the prompt.
def _promptForOutputDir(baseDir):
    candidates = sorted(d for d in glob.glob(os.path.join(baseDir, '*'))
                         if os.path.isdir(d) and '_backup_' not in os.path.basename(d)
                         and os.path.exists(os.path.join(d, 'latticeSamplerResults.txt')))
    if len(candidates) <= 1:
        return candidates[0] if candidates else os.path.join(baseDir, 'output')
    print('Multiple search output trees found - which one do you want to view?')
    for i, d in enumerate(candidates):
        print(f'  [{i}] {os.path.basename(d)}')
    choice = input(f'Enter a number (default 0 = {os.path.basename(candidates[0])}): ').strip()
    return candidates[int(choice)] if choice else candidates[0]


# Trajectories are saved per round (round_NNN/discoveryTrajectories/{label}.npz - see
# saveDiscoveryTrajectory), not in one flat directory, since the save cap is scoped per round. A
# label alone doesn't say which round found it, so search across all of them.
def _findDiscoveryNpz(label):
    matches = glob.glob(os.path.join(OUTPUT_DIR, 'round_*', 'discoveryTrajectories', f'{label}.npz'))
    if not matches:
        raise FileNotFoundError(f'No saved trajectory found for {label!r} in any round_*/discoveryTrajectories/ '
                                 f'(it may not have made that round\'s top-N cut, or the label is wrong)')
    if len(matches) > 1:
        raise RuntimeError(f'{label!r} matched multiple rounds: {matches} - this should not happen, labels are unique')
    return matches[0]


# Mirrors ODEClass.plotInteractive's exact styling (line width/colorscale/colorbar/scene layout),
# but reads from a saved discovery .npz (see engine.py's saveDiscoveryTrajectoriesBatch) instead
# of running a fresh integration - so a discovery's figure can be regenerated without re-running
# its stochastic (IC-dependent) trajectory.
def plotDiscoveryTrajectory(npzPathOrLabel, doSave=True, outDir=DEFAULT_PLOT_DIR):
    npzPath = npzPathOrLabel
    if not npzPath.endswith('.npz'):
        npzPath = _findDiscoveryNpz(npzPathOrLabel)
    label = os.path.splitext(os.path.basename(npzPath))[0]

    data = np.load(npzPath)
    t, x, y, z = data['t'], data['x'], data['y'], data['z']
    numTicks = 6
    tickvals = np.linspace(t[0], t[-1], numTicks)
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            width=2,
            color=t,
            colorscale="Plasma",
            cmin=t[0],
            cmax=t[-1],
            colorbar=dict(
                title='Time',
                tickmode='array',
                tickvals=tickvals,
                tickformat='.0f',
                ticklabeloverflow="allow",
            ),
        ),
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30),
        title=label,
    )

    if doSave:
        os.makedirs(outDir, exist_ok=True)
        outPath = os.path.join(outDir, f'{label}.html')
        fig.write_html(outPath)
        print(f'Saved to {outPath}')

    return fig


# Run standalone: `python3 geodesicBeamSearch/plotDiscoveryTrajectory.py` (no args) prompts for a
# label/path interactively; `python3 geodesicBeamSearch/plotDiscoveryTrajectory.py NewHit_42` (or a
# full .npz path) skips the prompt for scripting/repeat use.
if __name__ == '__main__':
    OUTPUT_DIR = _promptForOutputDir(SCRIPT_DIR)
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input(
            f'Discovery label (searched for across {OUTPUT_DIR}/round_*/discoveryTrajectories/) '
            f'or a path to a .npz file: '
        ).strip()
    plotDiscoveryTrajectory(target)
