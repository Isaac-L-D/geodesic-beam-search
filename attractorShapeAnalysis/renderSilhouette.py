import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Renders a trajectory to a clean silhouette image from several fixed camera angles - the input
# zernikeIcpCascade.py's Zernike descriptor is computed from. Multiple angles because a single
# fixed camera view biases the comparison toward whatever orientation two trajectories happen to
# share. No torch/torchvision import here - this pipeline only needs the rendered image, not a
# learned embedding of it, and importing a deep-learning framework would cost real startup time
# for nothing.

CAMERA_ANGLES = [(20, 30), (20, 120), (20, 210), (20, 300), (60, 45), (60, 225)]
IMAGE_SIZE = 224


# y is (3, N) - rows are x/y/z, columns are samples (ax.plot(y[0], y[1], y[2]) below is what
# fixes this convention). This is the OPPOSITE of attractorSimilarity.py's (N, 3) convention used
# everywhere else in this package - callers must transpose their (N,3) points before calling
# this, exactly once, at the boundary. Getting this backwards silently produces a degenerate
# few-point plot instead of an error - it has happened twice already in this project's history.
#
# linestyle='None' (markers only, no connecting line) rather than a connected line: at the full
# point count a trajectory needs to fill in a clean silhouette, a connected line draws a straight
# chord between every consecutive pair of points - harmless where the curve is locally smooth,
# but a chaotic trajectory folds back on itself constantly, so many of those chords cut straight
# across real folds instead of tracing them, corrupting the shape the descriptor sees. Markers
# can't produce that artifact at any density. Measured across the 36 known attractors: switching
# to markers-only raised self/cross AUC from 0.9735 to 0.9978, while cutting this function's
# memory footprint from ~1.18GB to ~32MB per call (matplotlib's 3D scatter was tried too - same
# artifact-free property, but ~20x slower due to mplot3d's per-point depth-sorting for a plain
# ax.plot call that never needed it).
def renderTrajectoryImage(y, elev, azim):
    fig = plt.figure(figsize=(3, 3), dpi=IMAGE_SIZE // 3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], linestyle='None', marker='.', markersize=1.2, color='black')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def renderAllViews(y, angles=CAMERA_ANGLES):
    return [renderTrajectoryImage(y, elev, azim) for elev, azim in angles]
