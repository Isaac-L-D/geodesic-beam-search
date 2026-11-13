import os
import sys
_realStdoutFd = os.dup(1)
_devnullFd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnullFd, 1)
os.close(_devnullFd)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'geodesicBeamSearch')
import engine as ls
from ODEClass import ODE
from jitcode import jitcode_lyap, y as jy

def out(s):
    os.write(_realStdoutFd, (s + "\n").encode())

PLOTS_DIR = 'Plots/classificationMethods'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- pick test candidates: a known-good baseline plus two hits from the earlier redraw test ---
labels, vecs = ls.loadKnownAttractorSeeds()
idx = {l: i for i, l in enumerate(labels)}
candidates = {'lorenz': ls.invertFlatten(vecs[idx['lorenz']])}

hitsDf = pd.read_csv('geodesicBeamSearch/output/allUniqueHits.csv')
coefCols = [c for c in hitsDf.columns if c.startswith('dx_') or c.startswith('dy_') or c.startswith('dz_')]
for label in ['NewHit_0', 'NewHit_2']:
    row = hitsDf[hitsDf['Label'] == label].iloc[0]
    candidates[label] = ls.invertFlatten(row[coefCols].values.astype(np.float64))


def buildGenericSystemFullSpec(nBasic=3):
    monomials = [1]
    for i in range(1, 27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        monomials.append((jy(0) ** d1) * (jy(1) ** d2) * (jy(2) ** d3))
    return monomials


# ---------- 1. Full Lyapunov spectrum + Kaplan-Yorke dimension ----------
def fullLyapunovSpectrum(WScaled, endTime=1000.0, renormTime=1.0, transientTime=50.0):
    monomials = buildGenericSystemFullSpec()
    dx = sum(WScaled[0, i] * monomials[i] for i in range(27))
    dy = sum(WScaled[1, i] * monomials[i] for i in range(27))
    dz = sum(WScaled[2, i] * monomials[i] for i in range(27))
    j = jitcode_lyap([dx, dy, dz], n_lyap=3, verbose=False)
    j.set_integrator('vode', method='bdf', atol=1e-6, rtol=1e-6)
    pos = np.random.uniform(-1, 1, 3)
    j.set_initial_value(pos)
    numTransientSteps = int(round(transientTime / renormTime))
    numSteps = int(round(endTime / renormTime))
    allLyaps = []
    for i in range(1, numTransientSteps + numSteps + 1):
        res = j.integrate(i * renormTime)
        allLyaps.append(res[1])
    allLyaps = np.vstack(allLyaps)[numTransientSteps:]
    spectrum = np.sort(allLyaps.mean(axis=0))[::-1]  # descending
    return spectrum


def kaplanYorkeDimension(spectrum):
    cumsum = 0.0
    for j in range(len(spectrum)):
        cumsum += spectrum[j]
        if cumsum < 0:
            if j == 0:
                return 0.0
            return j + (cumsum - spectrum[j]) / abs(spectrum[j])
    return float(len(spectrum))  # all exponents non-negative


# ---------- 2. Power spectrum ----------
def powerSpectrum(t, x, name):
    dt = t[1] - t[0]
    n = len(x)
    xCentered = x - x.mean()
    window = np.hanning(n)
    fft = np.fft.rfft(xCentered * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft) ** 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(freqs, power)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power (log scale)')
    ax.set_title(f'Power spectrum: {name}')
    ax.set_xlim(0, freqs[-1] * 0.2)
    fig.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/{name}_powerSpectrum.png', dpi=100)
    plt.close(fig)
    return freqs, power


# ---------- 3. Poincare section ----------
def poincareSection(y, name, axis=2):
    plane = y[axis].mean()
    crossings = []
    for i in range(len(y[axis]) - 1):
        if y[axis][i] < plane <= y[axis][i + 1]:
            frac = (plane - y[axis][i]) / (y[axis][i + 1] - y[axis][i])
            others = [k for k in range(3) if k != axis]
            pt = [y[k][i] + frac * (y[k][i + 1] - y[k][i]) for k in others]
            crossings.append(pt)
    crossings = np.array(crossings)
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(crossings) > 0:
        ax.scatter(crossings[:, 0], crossings[:, 1], s=2, alpha=0.5)
    ax.set_title(f'Poincare section: {name} (n={len(crossings)} crossings)')
    fig.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/{name}_poincare.png', dpi=100)
    plt.close(fig)
    return crossings


# ---------- 4. Correlation dimension (Grassberger-Procaccia) ----------
def correlationDimension(points, name, nSample=1500):
    if len(points) > nSample:
        idxs = np.random.choice(len(points), nSample, replace=False)
        points = points[idxs]
    from scipy.spatial.distance import pdist
    dists = pdist(points)
    dists = dists[dists > 0]
    rs = np.logspace(np.log10(np.percentile(dists, 1)), np.log10(np.percentile(dists, 50)), 20)
    counts = np.array([(dists < r).sum() for r in rs])
    total = len(dists)
    Cr = counts / total
    valid = Cr > 0
    logR = np.log(rs[valid])
    logC = np.log(Cr[valid])
    # fit slope over the middle portion (avoid saturation at both ends)
    mid = slice(len(logR) // 4, 3 * len(logR) // 4)
    slope, intercept = np.polyfit(logR[mid], logC[mid], 1)
    return slope, logR, logC


# ---------- 5. 0-1 test for chaos (Gottwald-Melbourne) ----------
# NOTE: still gives false negatives on known-chaotic systems (e.g. lorenz reads K=-0.02, should
# be near 1) even after fixing the Vosc bias-correction term below. The basic fixed-phase
# theta(j)=jc formulation used here is known to be unreliable for systems with strong
# quasi-periodic-looking structure like Lorenz - the literature's "modified" 0-1 test uses an
# adaptive phase theta(j)=jc+cumsum(x) instead, which would need a bigger rewrite. Don't trust
# this method's output as-is.
def zeroOneTest(x, nC=10, maxN=4000):
    # The Mn computation below is O(nCut*N) per c value - fine at a few thousand points, not at
    # hundreds of thousands. Subsample rather than scaling nCut with the full trajectory length.
    if len(x) > maxN:
        stride = len(x) // maxN
        x = x[::stride]
    N = len(x)
    nCut = N // 10
    # Correction term needs the squared MEAN of x, not the mean of x-squared - do not demean x
    # first, or this term silently becomes ~0 and the correction (meant to remove bias from a
    # non-zero mean) turns into subtracting the wrong quantity instead of doing nothing.
    xMean = x.mean()
    ks = []
    rng = np.random.default_rng(0)
    for _ in range(nC):
        c = rng.uniform(np.pi / 5, 4 * np.pi / 5)
        j = np.arange(1, N + 1)
        p = np.cumsum(x * np.cos(j * c))
        q = np.cumsum(x * np.sin(j * c))
        ns = np.arange(1, nCut + 1)
        Mn = np.array([np.mean((p[n:] - p[:-n]) ** 2 + (q[n:] - q[:-n]) ** 2) for n in ns])
        Vosc = xMean ** 2 * (1 - np.cos(ns * c)) / (1 - np.cos(c))
        Dn = Mn - Vosc
        K = np.corrcoef(ns, Dn)[0, 1]
        ks.append(K)
    return np.median(ks)


# ---------- run everything ----------
if __name__ == "__main__":
    results = []
    for name, W in candidates.items():
        out(f"\n=== {name} ===")
        tmp = ODE(W=W, doFitToCube=False)
        scaleBy = max(1, ls.EVAL_REFERENCE_NORM / np.linalg.norm(tmp.W))
        WScaled = tmp.W * scaleBy

        row = {'name': name}

        t0 = time.time()
        try:
            spectrum = fullLyapunovSpectrum(WScaled)
            kyDim = kaplanYorkeDimension(spectrum)
            dt1 = time.time() - t0
            out(f"full spectrum (native, /scaleBy): {spectrum/scaleBy} time={dt1:.1f}s")
            out(f"Kaplan-Yorke dimension: {kyDim:.3f}")
            row['spectrum_time'] = dt1
            row['ky_dim'] = kyDim
        except Exception as e:
            dt1 = time.time() - t0
            out(f"full spectrum FAILED: {e} time={dt1:.1f}s")
            row['spectrum_time'] = dt1
            row['ky_dim'] = None

        t0 = time.time()
        ode = ODE(W=WScaled)
        result = None
        for _ in range(5):  # NewHit_0-style candidates blow up from most ICs - retry before giving up
            pos = np.random.uniform(-1, 1, 3)
            result = ode.integrate(initial_pos=pos, endTime=2000.0)
            if result is not None:
                break
        dt2 = time.time() - t0
        if result is None:
            out(f"trajectory generation FAILED on all 5 attempts (blowup) - time={dt2:.1f}s, "
                f"skipping power spectrum/Poincare/correlation dim/0-1 test")
            row['traj_time'] = dt2
            row['traj_points'] = 0
            results.append(row)
            continue
        t, y = result
        out(f"trajectory generation: {len(t)} points, time={dt2:.1f}s")
        row['traj_time'] = dt2
        row['traj_points'] = len(t)

        # drop transient
        cut = len(t) // 5
        t, y = t[cut:], [yc[cut:] for yc in y]

        t0 = time.time()
        powerSpectrum(t, y[0], name)
        dt3 = time.time() - t0
        out(f"power spectrum + plot: time={dt3:.1f}s")
        row['power_time'] = dt3

        t0 = time.time()
        crossings = poincareSection(y, name)
        dt4 = time.time() - t0
        out(f"poincare section + plot: {len(crossings)} crossings, time={dt4:.1f}s")
        row['poincare_time'] = dt4
        row['poincare_crossings'] = len(crossings)

        t0 = time.time()
        try:
            points = np.column_stack(y)
            slope, logR, logC = correlationDimension(points, name)
            dt5 = time.time() - t0
            out(f"correlation dimension: {slope:.3f} time={dt5:.1f}s")
            row['corr_dim'] = slope
            row['corrdim_time'] = dt5
        except Exception as e:
            dt5 = time.time() - t0
            out(f"correlation dimension FAILED: {e} time={dt5:.1f}s")
            row['corr_dim'] = None
            row['corrdim_time'] = dt5

        t0 = time.time()
        k01 = zeroOneTest(y[0])
        dt6 = time.time() - t0
        out(f"0-1 test K statistic: {k01:.3f} time={dt6:.1f}s")
        row['zero_one_K'] = k01
        row['zeroone_time'] = dt6

        results.append(row)

    resultsDf = pd.DataFrame(results)
    out("\n=== SUMMARY ===")
    out(resultsDf.to_string())
    resultsDf.to_csv('attractorShapeAnalysis/classificationMethodsResults.csv', index=False)
    out("DONE")
