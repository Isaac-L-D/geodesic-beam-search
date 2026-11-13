import os
import sys
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ODEClass import ODE

RUN_DIR = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output')

# A hand-picked, visually diverse sample of real discoveries from the reported 15-round
# run - not a representative/random sample, chosen for Figure "discoveredAttractors" in
# the paper. displayLabel (Discovery A-F) is what appears in the published figure; the
# internal NewHit_ ID is kept here (and in the paper's caption) only for reproducibility.
# (elev, azim) tuned per-panel so each shape's distinguishing structure isn't foreshortened.
# Sorted by round number; Discovery labels A-F follow that same order.
#
# Each system here was independently verified to stay bounded under a from-saved-endpoint
# re-integration run 20x longer than the saved window, not just trusted from the chaos
# oracle's own classification - candidates that looked promising in preview but collapsed
# to a fixed point or blew up under that check were excluded.
SELECTED = [
    ('round_001', 'NewHit_68',    'Discovery A', 'Round 1',  35, -60),
    ('round_001', 'NewHit_60',    'Discovery B', 'Round 1',  20,  20),
    ('round_006', 'NewHit_3980',  'Discovery C', 'Round 6',  20, 250),
    ('round_008', 'NewHit_6143',  'Discovery D', 'Round 8',  30,  70),
    ('round_010', 'NewHit_7753',  'Discovery E', 'Round 10', 30,  45),
    ('round_012', 'NewHit_9756',  'Discovery F', 'Round 12', 30,  70),
]

# zoom=1.0 (matplotlib's own default margin) never clips regardless of a trajectory's
# shape or viewing angle - confirmed empirically across every angle used here, and is the
# default below. Some shapes tolerate more before clipping (checked individually, not
# assumed) - a thin diagonal shape like E's doesn't reach its own box's corners even at
# zoom=1.0, so it under-fills its panel at that default; raising it per-discovery for just
# those cases fills the panel properly from the render itself, rather than leaning entirely
# on the crop/upscale step after (which can't fix emptiness *within* a tile's own bounding
# box, only normalize bounding-box size *across* tiles).
DEFAULT_ZOOM = 1.0
ZOOM_OVERRIDES = {
    'NewHit_7753': 1.6,  # Discovery E - thin diagonal shape, doesn't reach its box corners at 1.0
}


# High source DPI so even the panel with the smallest content fill (a trajectory that only
# occupies a modest fraction of its own bounding cube at this viewing angle) still has ample
# pixels after its tight crop - target size below is the LARGEST crop (so smaller-filled
# panels scale UP to match it, not the other way around, matching what "evenly sized" means
# visually: everything reads as filling its cell about as well as the best-filled one does),
# and this DPI keeps every such upscale factor modest enough that LANCZOS resampling doesn't
# introduce visible blur.
RENDER_DPI = 450


def renderPanelToImage(ode, d, elev, azim, title, zoom):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ode.plotStaticGradient(d['x'], d['y'], d['z'], ax=ax, elev=elev, azim=azim, linewidth=0.5, title=title)
    ax.set_box_aspect(None, zoom=zoom)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=RENDER_DPI, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# Tight bounding box of non-white content, with a small uniform pad so lines right at the
# true edge don't look clipped after crop/resize - same "content" area in every tile
# afterward regardless of the panel's own data shape, which is what makes padding-to-a-
# shared-size next produce visually comparable tiles.
def tightCropWhitespace(img, padPx=15):
    arr = np.asarray(img)
    nonWhite = (arr < 250).any(axis=2)
    rows = np.where(nonWhite.any(axis=1))[0]
    cols = np.where(nonWhite.any(axis=0))[0]
    top, bottom = max(rows.min() - padPx, 0), min(rows.max() + padPx, arr.shape[0])
    left, right = max(cols.min() - padPx, 0), min(cols.max() + padPx, arr.shape[1])
    return img.crop((left, top, right, bottom))


# "Contain" scaling: resize the tile (preserving its own aspect ratio, never stretching it)
# until it fits exactly inside targetSize, using LANCZOS - a high-quality filter that stays
# sharp for the modest up- or down-scale factors every tile needs here - then center it on a
# white canvas of exactly targetSize.
def scaleAndPadToSize(img, targetSize):
    scale = min(targetSize[0] / img.width, targetSize[1] / img.height)
    newSize = (round(img.width * scale), round(img.height * scale))
    resized = img.resize(newSize, Image.LANCZOS)
    canvas = Image.new('RGB', targetSize, 'white')
    x = (targetSize[0] - resized.width) // 2
    y = (targetSize[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas, scale


# Rendered directly at (width, height) in pixels (via figsize=pixels/dpi) rather than
# rendered small and resized up afterward - a resize-to-enlarge is exactly the upscaling
# that introduces blur, so the colorbar is sized correctly from the start instead. Uses the
# same DPI as the attractor panels (RENDER_DPI) and a font size chosen to match their title
# text's physical size at that DPI - otherwise a small font at a lower DPI renders tiny and
# cramped next to the (much higher-DPI, larger-canvas) panels, as happened before this fix.
def makeColorbarStrip(width, height, cmap='plasma'):
    dpi = RENDER_DPI
    fontsize = 34
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    # ax narrower and shifted left of its earlier position, leaving more room to its right
    # for the rotated axis label - at the old width/position the label's own height (as
    # drawn sideways) ran past the right edge of this strip and got cut off. Bottom/top
    # margins (0.16 each, vs. the earlier 0.1/0.12) leave enough room for the "Start"/"End"
    # text below/above the bar itself to not run past the canvas edge - the earlier, tighter
    # margin clipped the top of "Start".
    ax = fig.add_axes([0.25, 0.16, 0.16, 0.68])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cm.get_cmap(cmap), norm=norm)
    cb.set_ticks([])
    # "Start"/"End" placed above/below the bar itself (0=bottom=Start, 1=top=End) rather than
    # as side tick labels. The axis title (what the bar represents) uses set_label, not manual
    # text placement - it rotates and positions itself safely beside a vertical colorbar
    # without needing hand-tuned coordinates that can run outside the canvas.
    ax.text(0.5, -0.04, 'Start', transform=ax.transAxes, ha='center', va='top', fontsize=fontsize)
    ax.text(0.5, 1.04, 'End', transform=ax.transAxes, ha='center', va='bottom', fontsize=fontsize)
    cb.set_label('Progress along trajectory', fontsize=fontsize, labelpad=15)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    assert img.width == width and img.height == height, (img.width, img.height, width, height)
    return img


def main():
    ode = ODE()
    tiles = []
    for roundDir, label, displayLabel, roundLabel, elev, azim in SELECTED:
        npzPath = os.path.join(RUN_DIR, roundDir, 'discoveryTrajectories', f'{label}.npz')
        d = np.load(npzPath)
        title = (f'{displayLabel} ({roundLabel}), LLE $\\approx$ {float(d["lleMean"]):.2f}\n'
                 f'Elevation={elev}$^\\circ$, Azimuth={azim}$^\\circ$')
        zoom = ZOOM_OVERRIDES.get(label, DEFAULT_ZOOM)
        img = renderPanelToImage(ode, d, elev, azim, title, zoom)
        tiles.append(tightCropWhitespace(img))
        print(f'{displayLabel} = {label} ({roundDir})')

    # Largest crop in each dimension - every tile scales to look about as filled as the
    # best-filled one, rather than all being capped down to the smallest.
    targetW = max(t.width for t in tiles)
    targetH = max(t.height for t in tiles)
    scaled = [scaleAndPadToSize(t, (targetW, targetH)) for t in tiles]
    tiles = [s[0] for s in scaled]
    for (_, label, displayLabel, *_), (_, scale) in zip(SELECTED, scaled):
        print(f'{displayLabel}: scale factor {scale:.2f}x ({"upscaled" if scale > 1 else "downscaled" if scale < 1 else "unchanged"})')

    gap = 24
    nCols, nRows = 3, 2
    cbarW = round(targetW * 0.22)
    gridW = nCols * targetW + (nCols - 1) * gap
    gridH = nRows * targetH + (nRows - 1) * gap
    canvas = Image.new('RGB', (gridW + cbarW, gridH), 'white')
    for i, tile in enumerate(tiles):
        r, c = divmod(i, nCols)
        x = c * (targetW + gap)
        y = r * (targetH + gap)
        canvas.paste(tile, (x, y))
    cbar = makeColorbarStrip(cbarW, gridH)
    canvas.paste(cbar, (gridW, 0))

    outPath = os.path.join(SCRIPT_DIR, 'figures', 'discoveredAttractors.png')
    canvas.save(outPath)
    print(f'Saved {outPath}')


if __name__ == '__main__':
    main()