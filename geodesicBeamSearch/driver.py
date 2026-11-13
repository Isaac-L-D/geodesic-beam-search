import os
import sys
import shutil
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from wakepy import keep

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import engine as ls

# How many of a round's new, distinct, chaotic discoveries get carried forward as the NEXT
# round's seeds. Unlike the earlier uncapped variant of this search (which seeded every new
# discovery, causing the seed count - and therefore compute - to grow every round), this caps
# the seed set to the TOP_N_SEEDS most chaotic (highest LLEMean) discoveries each round, so
# per-round compute stays roughly constant instead of compounding.
# Novelty checking (referenceDescriptors) and the backward/sideways catalog filter
# (catalogVecs) are NOT capped - every discovery ever made, seeded or not, still counts toward
# both, so discoveries that don't become seeds still block future re-exploration of their
# territory and still count as "already known" for distinctness checks.
TOP_N_SEEDS = 100

# This variant writes to its own output tree - its own log, checkpoints, unique-hits CSV, and
# round directories - so it never touches or depends on the earlier uncapped variant's progress.
# It always starts fresh (round 1, known-attractor seeds) unless a checkpoint already exists in
# ITS OWN checkpoint dir, in which case it resumes from that the same way.
ls.OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(ls.OUTPUT_DIR, exist_ok=True)
ls.RESULTS_LOG_PATH = os.path.join(ls.OUTPUT_DIR, 'latticeSamplerResults.txt')
ls.ALL_UNIQUE_HITS_PATH = os.path.join(ls.OUTPUT_DIR, 'allUniqueHits.csv')
ls.TAPER_PLOT_PATH = os.path.join(ls.OUTPUT_DIR, 'newHitsPerRound.png')
ls.CHECKPOINT_DIR = os.path.join(ls.OUTPUT_DIR, 'checkpoints')
os.makedirs(ls.CHECKPOINT_DIR, exist_ok=True)


# Ranks `labels` by the LLEMean recorded for them in roundDirForLLEData's
# latticeSamplerUniqueHits.csv (written by ls.runRound for the round that produced them) and
# keeps only the top n. Returns everything unchanged (plus the mean LLE of the whole set) if
# there's nothing to trim. The 4th return value - mean LLE of whichever labels actually get
# carried forward as next round's seeds - feeds the diagnostic plots below.
def capSeedsToTopN(roundDirForLLEData, labels, vecs, failsafePoints, n):
    labels = np.asarray(labels)
    vecs = np.asarray(vecs)
    hitsPath = os.path.join(roundDirForLLEData, 'latticeSamplerUniqueHits.csv')
    lleByLabel = pd.read_csv(hitsPath, usecols=['Label', 'LLEMean']).set_index('Label')['LLEMean']
    if len(labels) <= n:
        return labels, vecs, failsafePoints, float(lleByLabel[labels].mean())
    order = np.argsort([-lleByLabel[label] for label in labels])[:n]
    topLabels = labels[order]
    topVecs = vecs[order]
    topFailsafePoints = {label: failsafePoints[label] for label in topLabels}
    return topLabels, topVecs, topFailsafePoints, float(lleByLabel[topLabels].mean())


# --- Round-level diagnostic plots ------------------------------------------------------------
# Persisted separately from ls.saveCheckpoint's pickle (which has a fixed schema shared with
# the earlier uncapped variant) so this stays self-contained to this file - a plain CSV, rewritten in full
# each round, that survives a resume the same way roundCounts does.
DIAGNOSTICS_CSV_PATH = os.path.join(ls.OUTPUT_DIR, 'roundDiagnostics.csv')
DIAGNOSTICS_PLOT_PATH = os.path.join(ls.OUTPUT_DIR, 'roundDiagnostics.png')


def loadRoundStats():
    if not os.path.exists(DIAGNOSTICS_CSV_PATH):
        return []
    return pd.read_csv(DIAGNOSTICS_CSV_PATH).to_dict('records')


# Everything here is reconstructed from files runRound already writes (successfulInitialPositions
# .csv has an IsChaotic column, so hits = its sum; newLabels/topSeedsMeanLLE are already in hand
# at the call site) - no changes needed to engine.py itself. The one assumption: candidates
# = seedsThisRound * N_DIRECTIONS_PER_SEED * len(RADII), i.e. the backward/sideways catalog filter
# rejects nothing - true in all 5 rounds observed so far (every round's log shows "0 of N thrown
# out"). If that ever stops being exactly 0%, this will overcount candidates relative to what
# actually got evaluated; the true post-filter count is only available by parsing the "N candidates
# after backward/sideways catalog filtering" line out of latticeSamplerResults.txt, which wasn't
# worth the fragility unless/until this assumption actually breaks.
def collectRoundDiagnostics(roundNum, seedsThisRound, roundDir, newLabels, topSeedsMeanLLE):
    successPath = os.path.join(roundDir, 'successfulInitialPositions.csv')
    successDf = pd.read_csv(successPath) if os.path.exists(successPath) else pd.DataFrame(columns=['IsChaotic'])
    success = len(successDf)
    hits = int(successDf['IsChaotic'].sum()) if success else 0
    candidates = seedsThisRound * ls.N_DIRECTIONS_PER_SEED * len(ls.RADII)
    return {
        'Round': roundNum, 'Seeds': seedsThisRound, 'Candidates': candidates,
        'Success': success, 'Hits': hits, 'Distinct': len(newLabels),
        'TopSeedsMeanLLE': topSeedsMeanLLE,
    }


# Bar = total (denominator), shaded portion = positive (numerator), stacked so the boundary
# between them sits exactly at the positive count - a real, readable value on the same count axis,
# not a fabricated position on a second scale. The exact fraction is added as a text label at that
# same point, so the percentage is precise without needing its own axis at all.
def _stackedFractionPanel(ax, rounds, positive, total, positiveLabel, totalLabel, title):
    positive = positive.astype(float)
    total = total.astype(float)
    negative = total - positive
    ax.bar(rounds, positive, color='#2a78d6', label=positiveLabel)
    ax.bar(rounds, negative, bottom=positive, color='#c7d3e0', label=f'{totalLabel} (rest)')
    frac = np.divide(positive, total, out=np.full(len(total), np.nan), where=total > 0)
    ax.plot(rounds, positive, color='#16324f', marker='o', markersize=5, linewidth=1.5, zorder=5)
    for r, p, f in zip(rounds, positive, frac):
        if np.isfinite(f):
            ax.annotate(f'{f * 100:.1f}%', (r, p), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=8, color='#16324f')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Round')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(rounds)


# distinct/seeds isn't a part-of-a-whole fraction (distinct can exceed seeds), so it doesn't fit
# the stacked-bar pattern above - shown instead as paired bars (seeds vs. distinct, same count
# axis) with the ratio itself as a text label, same "real axis + exact text label" principle.
def _yieldRatioPanel(ax, rounds, seeds, distinct):
    seeds = seeds.astype(float)
    distinct = distinct.astype(float)
    ax.bar(rounds - 0.2, seeds, width=0.4, color='#c7d3e0', label='Seeds (in)')
    ax.bar(rounds + 0.2, distinct, width=0.4, color='#2a78d6', label='Distinct (out)')
    ratio = np.divide(distinct, seeds, out=np.full(len(seeds), np.nan), where=seeds > 0)
    for r, d, s, ra in zip(rounds, distinct, seeds, ratio):
        if np.isfinite(ra):
            ax.annotate(f'×{ra:.2f}', (r, max(d, s)), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=8, color='#16324f')
    ax.set_title('Yield Ratio: Distinct Discoveries per Seed', fontsize=11)
    ax.set_xlabel('Round')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(rounds)


def _meanLLEPanel(ax, rounds, meanLLE):
    ax.plot(rounds, meanLLE, color='#2a78d6', marker='o')
    ax.set_title(f'Mean LLE of top {TOP_N_SEEDS} seeds carried forward', fontsize=11)
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean LLE')
    ax.set_xticks(rounds)


# Rewrites both the CSV (source of truth, resumable - see loadRoundStats) and the PNG in full
# every round, same pattern as ls.saveTaperPlot. Uses ls.plt rather than importing matplotlib
# separately here - engine.py already did matplotlib.use('Agg') at import time.
def saveDiagnosticPlots(roundStats):
    statsDf = pd.DataFrame(roundStats)
    statsDf.to_csv(DIAGNOSTICS_CSV_PATH, index=False)
    if statsDf.empty:
        return

    rounds = statsDf['Round'].to_numpy()
    # 2x6 grid rather than 2x3: the bottom row only ever has 2 real panels (a 3rd would have
    # no natural metric to show), so each of those 2 spans 3 of 6 columns - filling the row's
    # full width - while the top row's 3 panels each span 2 of 6 columns, instead of leaving a
    # dead empty cell where a 3rd bottom panel would otherwise go.
    fig = ls.plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 6)
    axTopLeft, axTopMid, axTopRight = (fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
                                        fig.add_subplot(gs[0, 4:6]))
    axBotLeft, axBotRight = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
    _meanLLEPanel(axTopLeft, rounds, statsDf['TopSeedsMeanLLE'].to_numpy(dtype=float))
    _stackedFractionPanel(axTopMid, rounds, statsDf['Success'].to_numpy(), statsDf['Candidates'].to_numpy(),
                          'Success', 'Candidates', 'Success Fraction (Success / Candidates)')
    _stackedFractionPanel(axTopRight, rounds, statsDf['Hits'].to_numpy(), statsDf['Success'].to_numpy(),
                          'Chaotic hits', 'Success', 'Hit Fraction (Hits / Success)')
    _stackedFractionPanel(axBotLeft, rounds, statsDf['Distinct'].to_numpy(), statsDf['Hits'].to_numpy(),
                          'Distinct', 'Hits', 'Novelty Fraction (Distinct / Hits)')
    _yieldRatioPanel(axBotRight, rounds, statsDf['Seeds'].to_numpy(), statsDf['Distinct'].to_numpy())
    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_PLOT_PATH, dpi=150, bbox_inches='tight')
    ls.plt.close(fig)


def main():
    ls.cleanupStaleJitcodeTempDirs()
    threading.Thread(target=ls._cleanupLoop, daemon=True).start()
    rng = np.random.default_rng(ls.RANDOM_STATE)

    checkpoint = ls.loadLatestCheckpoint()
    if checkpoint is not None:
        roundNum = checkpoint['roundNum']
        seedLabels = np.array(checkpoint['seedLabels'])
        seedVecs = checkpoint['seedVecs']
        catalogVecs = checkpoint['catalogVecs']
        referenceDescriptors = checkpoint['referenceDescriptors']
        newHitCount = checkpoint['newHitCount']
        nextSeed = checkpoint['nextSeed']
        knownFailsafePoints = checkpoint['knownFailsafePoints']
        roundCounts = checkpoint['roundCounts']
        ls.logResult(f"\nResuming from checkpoint: round {roundNum}, {len(seedLabels)} seeds, "
                     f"{newHitCount} hits found across earlier rounds.")
        staleRoundDir = os.path.join(ls.OUTPUT_DIR, f'round_{roundNum:03d}')
        if os.path.isdir(staleRoundDir):
            backupPath = staleRoundDir + '_interrupted_partial'
            shutil.move(staleRoundDir, backupPath)
            ls.logResult(f"Moved round {roundNum}'s interrupted partial output aside to {backupPath}")
            ls.moveRoundRowsFromCumulativeCsv(roundNum, backupPath)
    else:
        known = ls.loadDatasetAttractors()
        knownNames = [n for n in known if n not in ls.EXCLUDED_ATTRACTOR_NAMES]
        descTasks = [(('known', name), known[name], ls.RANDOM_STATE + 10 + i, None)
                     for i, name in enumerate(knownNames)]
        # maxtasksperchild=1 - see the same fix in engine.py's runRound for why.
        with Pool(initializer=ls._silenceFortranStdout, maxtasksperchild=1) as pool:
            knownResults = dict(tqdm(pool.imap_unordered(ls._descriptorTask, descTasks), total=len(descTasks),
                                      desc='Descriptors for known attractors'))
        referenceDescriptors = {name: knownResults[('known', name)] for name in knownNames
                                 if knownResults[('known', name)] is not None
                                 and knownResults[('known', name)]['curvature'] is not None
                                 and knownResults[('known', name)]['d2'] is not None}
        nextSeed = ls.RANDOM_STATE + 10 + len(knownNames)

        seedLabels, seedVecs = ls.loadKnownAttractorSeeds()
        catalogVecs = seedVecs.copy()
        newHitCount = 0
        roundNum = 1
        roundCounts = []
        knownFailsafePoints = {}

    # Independent of the checkpoint pickle above (see DIAGNOSTICS_CSV_PATH) - reloaded here either
    # way, resume or fresh start, since it's a no-op (empty list) when the file doesn't exist yet.
    roundStats = loadRoundStats()

    while True:
        seedsThisRound = len(seedLabels)  # captured before capSeedsToTopN reassigns seedLabels below
        newLabels, newVecs, newHitCount, nextSeed, newFailsafePoints = ls.runRound(
            roundNum, seedLabels, seedVecs, catalogVecs, referenceDescriptors, newHitCount,
            nextSeed, rng, knownFailsafePoints)
        roundCounts.append((roundNum, len(newLabels)))
        ls.saveTaperPlot(roundCounts)
        thisRoundDir = os.path.join(ls.OUTPUT_DIR, f'round_{roundNum:03d}')
        # Full discovery history, uncapped: every new hit still guards its own territory against
        # backward/sideways re-exploration next round, whether or not it becomes a seed.
        if len(newVecs):
            catalogVecs = np.vstack([catalogVecs, newVecs])
        if len(newLabels) == 0:
            roundStats.append(collectRoundDiagnostics(roundNum, seedsThisRound, thisRoundDir, newLabels, None))
            saveDiagnosticPlots(roundStats)
            ls.logResult(f"\nRound {roundNum} found no new unique attractors anywhere - stopping.")
            break

        seedLabels, seedVecs, knownFailsafePoints, topSeedsMeanLLE = capSeedsToTopN(
            thisRoundDir, newLabels, newVecs, newFailsafePoints, TOP_N_SEEDS)
        if len(seedLabels) < len(newLabels):
            ls.logResult(f"Using the top {len(seedLabels)} most chaotic of {len(newLabels)} new "
                         f"discoveries as round {roundNum + 1}'s seeds (every discovery still "
                         f"counts for novelty/catalog checks, capping only affects seeding).")

        roundStats.append(collectRoundDiagnostics(roundNum, seedsThisRound, thisRoundDir, newLabels, topSeedsMeanLLE))
        saveDiagnosticPlots(roundStats)

        roundNum += 1
        ls.saveCheckpoint(roundNum, seedLabels, seedVecs, catalogVecs, referenceDescriptors,
                          newHitCount, nextSeed, knownFailsafePoints, roundCounts)

    print(f"\nDone after {roundNum} round(s). Log saved to {ls.RESULTS_LOG_PATH}, "
          f"taper plot saved to {ls.TAPER_PLOT_PATH}, "
          f"all unique hits saved to {ls.ALL_UNIQUE_HITS_PATH}")
    ls.cleanupJitcodeTempDirsUnconditional()


if __name__ == '__main__':
    try:
        with keep.presenting():
            main()
    except KeyboardInterrupt:
        ls.cleanupJitcodeTempDirsUnconditional()
        raise
