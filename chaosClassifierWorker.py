import os
import sys

# Must run before any other import: DVODE/LSODA write raw diagnostics straight to fd 1,
# which would corrupt the JSON result below. Redirect fd 1 to /dev/null and write JSON
# through a preserved duplicate instead.
_realStdoutFd = os.dup(1)
_devnullFd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnullFd, 1)
os.close(_devnullFd)

import json
import random
import numpy as np
from ODEClass import ODE, logDetails

# DIMENSION_FLOOR separates curve-like/degenerate geometry (dimension ~1, e.g. limit cycles -
# the dominant failure mode found in the existing catalog) from genuine 3D chaotic flows (which
# must exceed 2 by Kaplan-Yorke, but a floor of 1.25 leaves margin for weakly-chaotic real
# attractors that clearly aren't curve-like - several known attractors sit at 1.3-1.5).
# DIMENSION_STABILITY_MAX_STD catches candidates whose geometry isn't reproducible across
# independent draws (genuine multistability/basin-dependence, not measurement noise). Set from
# validation against the 45 known-attractor catalog, integrating for the same endTime/
# renormTime/transientTime (divided by scaleBy) this worker uses: robustly chaotic known
# attractors (e.g. rucklidge) can show std up to ~0.19 across independent draws purely from this
# integration window's length, while genuinely unstable candidates (sprotttorus, pehlivanwei)
# sit at 0.33-0.55+ - 0.20 sits in the gap between those two populations.
DIMENSION_FLOOR = 1.25
DIMENSION_STABILITY_MAX_STD = 0.20
LLE_PASS_RATE_REQUIRED = 0.6

BASELINE_DRAWS = 3
MAX_DRAWS = 5
# Escalation triggers on the baseline are deliberately more eager than the final cutoffs, since a
# std/mean computed from only BASELINE_DRAWS points is itself a noisy estimate of the true spread.
# BASELINE_DRAWS=2 was tried first and missed a known-bimodal candidate whose first two draws
# happened to land in the same regime by chance - 3 was needed to catch it reliably.
ESCALATE_CD_STD_MARGIN = 0.06
ESCALATE_CD_MEAN_MARGIN = 0.15


def _writeResult(obj):
    os.write(_realStdoutFd, json.dumps(obj).encode())


# Runs as a standalone subprocess so subprocess.run(timeout=...) can kill and reclaim it
# cleanly if it hangs, without the corruption risk of an in-process signal-based timeout.
def main():
    payload = json.loads(sys.stdin.read())
    if "seed" in payload:
        random.seed(payload["seed"])
    W = np.array(payload["W"], dtype=np.float64).reshape(3, 27)
    scaleBy = payload["scaleBy"]
    ode = ODE(W=W * scaleBy)

    status = logDetails()
    lleDraws, cdDraws = [], []

    def draw():
        lle, cd = ode.computeChaosSignals(
            endTime=1000.0 / scaleBy, renormTime=1.0 / scaleBy,
            transientTime=50.0 / scaleBy, logStatus=status)
        if lle is not None:
            lleDraws.append(lle / scaleBy)
        if cd is not None:
            cdDraws.append(cd)

    for _ in range(BASELINE_DRAWS):
        draw()

    baselinePassRate = (np.array(lleDraws) > 0).mean() if lleDraws else 0.0
    baselineStd = float(np.std(cdDraws)) if len(cdDraws) > 1 else None
    baselineMean = float(np.mean(cdDraws)) if cdDraws else None

    disagree = (baselinePassRate not in (0.0, 1.0)) or \
               (baselineStd is not None and baselineStd > ESCALATE_CD_STD_MARGIN) or \
               (baselineMean is not None and abs(baselineMean - DIMENSION_FLOOR) < ESCALATE_CD_MEAN_MARGIN)

    escalated = disagree
    if disagree:
        for _ in range(MAX_DRAWS - BASELINE_DRAWS):
            draw()

    # float(...) matters here, not just style: numpy.float64 >= LLE_PASS_RATE_REQUIRED below
    # produces numpy.bool_, and when that's falsy it propagates through the `and` chain into
    # isChaotic via short-circuit, which json.dumps can't serialize - crashing this subprocess
    # for the (common) non-chaotic case, silently discarding real classification detail. See
    # cdMean/cdStd two lines down, which already cast for the same reason - this one was missed.
    llePassRate = float((np.array(lleDraws) > 0).mean()) if lleDraws else 0.0
    lleMean = float(np.mean(lleDraws)) if lleDraws else None
    cdMean = float(np.mean(cdDraws)) if cdDraws else None
    cdStd = float(np.std(cdDraws)) if cdDraws else None

    lleOk = llePassRate >= LLE_PASS_RATE_REQUIRED
    dimFloorOk = (cdMean is not None) and (cdMean > DIMENSION_FLOOR)
    dimStableOk = (cdStd is not None) and (cdStd <= DIMENSION_STABILITY_MAX_STD)
    isChaotic = lleOk and dimFloorOk and dimStableOk

    # reasonsNotChaotic and the raw per-draw lists (not just their mean/std) are kept for
    # auditability - which specific check(s) a candidate failed, and what every individual draw
    # actually returned, not just a summary statistic.
    reasonsNotChaotic = []
    if not lleOk: reasonsNotChaotic.append(f'lle_pass_rate={llePassRate:.2f}')
    if not dimFloorOk: reasonsNotChaotic.append(f'cd_mean={cdMean}')
    if not dimStableOk: reasonsNotChaotic.append(f'cd_std={cdStd}')

    _writeResult({
        "status": status.status,
        "isChaotic": isChaotic,
        "nDraws": len(lleDraws),
        "escalated": escalated,
        "llePassRate": llePassRate,
        "lleMean": lleMean,
        "cdMean": cdMean,
        "cdStd": cdStd,
        "lleDraws": lleDraws,
        "cdDraws": cdDraws,
        "reasonsNotChaotic": reasonsNotChaotic,
    })


if __name__ == "__main__":
    main()
