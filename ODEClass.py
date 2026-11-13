import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import math
import sys
import csv
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.io as pio
import os
import glob
import random
import scipy.spatial.distance as dist
import psutil
import sympy as sp
from jitcode import jitcode_lyap
from jitcode import y as jy

# The 81 W coefficients are control_pars (set_parameters()) instead of baked into the compiled
# code, so compiling is a one-time project-wide cost, cached to disk, not a per-candidate one.
_LYAP_PAR_SYMBOLS = sp.symbols('p0:81')
_LYAP_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jitcodeCompiledCache')
_LYAP_CACHE_BASENAME = os.path.join(_LYAP_CACHE_DIR, 'lyapGenericCubic')


def _buildGenericLyapSystem():
    monomials = [1]
    for i in range(1, 27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        monomials.append((jy(0) ** d1) * (jy(1) ** d2) * (jy(2) ** d3))
    dx = sum(_LYAP_PAR_SYMBOLS[i] * monomials[i] for i in range(27))
    dy = sum(_LYAP_PAR_SYMBOLS[27 + i] * monomials[i] for i in range(27))
    dz = sum(_LYAP_PAR_SYMBOLS[54 + i] * monomials[i] for i in range(27))
    return [dx, dy, dz]


def _compileOwnLyap():
    os.makedirs(_LYAP_CACHE_DIR, exist_ok=True)
    j = jitcode_lyap(_buildGenericLyapSystem(), n_lyap=1, control_pars=_LYAP_PAR_SYMBOLS, verbose=False)
    j.generate_jac_sym(simplify=False)
    j.generate_f_C(do_cse=True)
    j.generate_jac_C(do_cse=True)
    j.save_compiled(destination=f'{_LYAP_CACHE_BASENAME}_{os.getpid()}', overwrite=True)
    # 2nd positional arg is named `interpolate` here but forwards into jitcode's `nsteps` slot.
    j.set_integrator('vode', 10000, method='bdf', atol=1e-6, rtol=1e-6)
    return j


def _tryLoadLyap(path):
    try:
        j = jitcode_lyap(n=3, n_lyap=1, control_pars=_LYAP_PAR_SYMBOLS, module_location=path, verbose=False)
        j.set_integrator('vode', 10000, method='bdf', atol=1e-6, rtol=1e-6)
        return j
    except Exception:
        return None


def _newLyapIntegrator():
    for path in glob.glob(_LYAP_CACHE_BASENAME + '_*.so'):
        j = _tryLoadLyap(path)
        if j is not None:
            return j
    return _compileOwnLyap()


# Grassberger-Procaccia correlation dimension - Grassberger & Procaccia, "Characterization of
# strange attractors," Physical Review Letters 50(5):346, 1983; and "Measuring the strangeness
# of strange attractors," Physica D 9(1):189-208, 1983. Measures the fractal geometry of the
# attractor, independent of and complementary to the Lyapunov exponent (which measures
# sensitivity to initial conditions, not geometric shape). A single LLE>0 draw was found
# unreliable near the chaos threshold (redraws flip sign for weakly chaotic/borderline
# candidates) - this dimension check catches cases LLE alone can't, e.g. candidates with
# consistently positive LLE whose attracting set is nevertheless curve-like (dimension ~1, a
# limit cycle) rather than a genuine fractal strange attractor. Operates directly on the true
# state trajectory (already available from integration) rather than a delay-embedding
# reconstruction from a single observable, since the delay-embedding step exists specifically
# for when full state access isn't available - not the case here. Fits the log-log slope of the
# correlation sum C(r) over the middle half of the sampled radius range (avoids under-sampling
# at small r and saturation at large r).
def correlationDimension(points, nSample=1500, rng=None):
    rng = rng or np.random
    points = np.asarray(points)
    if len(points) > nSample:
        idxs = rng.choice(len(points), nSample, replace=False)
        points = points[idxs]
    distances = dist.pdist(points)
    distances = distances[distances > 0]
    rs = np.logspace(np.log10(np.percentile(distances, 1)), np.log10(np.percentile(distances, 50)), 20)
    counts = np.array([(distances < r).sum() for r in rs])
    Cr = counts / len(distances)
    valid = Cr > 0
    logR, logC = np.log(rs[valid]), np.log(Cr[valid])
    mid = slice(len(logR) // 4, 3 * len(logR) // 4)
    slope, _ = np.polyfit(logR[mid], logC[mid], 1)
    return slope

# Symbolic derivatives of the 27-monomial basis (x^a*y^b*z^c), built once at import time since
# the basis itself never depends on W - only the per-instance W @ (these derivatives) does.
_symX, _symY, _symZ = sp.symbols('x y z')
_symMonomials = [_symX**(i // 9) * _symY**((i // 3) % 3) * _symZ**(i % 3) for i in range(27)]
_monomialJacobianExpr = sp.Matrix(_symMonomials).jacobian([_symX, _symY, _symZ])  # (27,3)
_monomialHessianExpr = [sp.hessian(m, [_symX, _symY, _symZ]) for m in _symMonomials]  # 27 x (3,3)
_monomialJacobianFunc = sp.lambdify((_symX, _symY, _symZ), _monomialJacobianExpr, 'numpy')
_monomialHessianFunc = sp.lambdify((_symX, _symY, _symZ), _monomialHessianExpr, 'numpy')
_monomialFunc = sp.lambdify((_symX, _symY, _symZ), sp.Matrix(_symMonomials), 'numpy')

class logDetails: #Dummy class to allow for logs to be passed from integrate() without taking up the return and complicating the code
    def __init__(self):
        self.status = "None"
        self.minDist = None
class ODE:
    doDefaultFitToCube = False
    fitToUnitCube = doDefaultFitToCube
    capIntegrationTime=False
    # When fitToUnitCube shrinks W, dynamics slow down by scaleFactor, so endTime is
    # stretched to compensate; this caps how much that stretch can grow endTime by.
    capScaleFactorIncrease = True
    maxScaleFactorIncrease = 10
    maxIntegrationTime=300.0
    atol = 1e-6
    jcMonomials = [1]
    for i in range(1,27):  
        tmpD1 = i // 9
        tmpD2 = (i // 3) % 3
        tmpD3 = i % 3
        jcMonomials.append((jy(0)**tmpD1)*(jy(1)**tmpD2)*(jy(2)**tmpD3))
    jcOdeSolver = "vode"
    jcMethodParam = 'bdf'
    jcEndTime = round(1e6)
    # computeLargestLyapunovExponent aborts (returning None) if the process's resident
    # memory exceeds this while running, so a pathological trajectory can't OOM a long-running caller.
    maxLyapunovMemoryMB = 2048
    # Exponents of x,y,z in each of the 27 monomials (column i encodes x^(i//9)*y^((i//3)%3)*z^(i%3)).
    # Precomputed once since computeJacobian's per-monomial exponents don't depend on W.
    monomialExponents = np.stack(
        [np.arange(27) // 9, (np.arange(27) // 3) % 3, np.arange(27) % 3], axis=1
    ).astype(np.float64)
    #List taken from https://www.dynamicmath.xyz/strange-attractors/, excluding those with terms with 3rd power or more variables
    def fitWToCube(self,W):
            # Uniformly rescales W so its largest-magnitude coefficient becomes 0.5,
            # keeping trajectories from blowing up while preserving relative dynamics.
            maxVal = W.max()
            minVal = W.min()
            absMax = max(abs(minVal),abs(maxVal))
            scaleFactor = 0.5*(1/absMax)
            return (W.copy()*scaleFactor),scaleFactor
    def loadStrangeAttractors(self):
        tmpDict ={}
        with open('selectedStrangeAttractors.csv',mode="r",newline='') as csvFile:
            reader = csv.reader(csvFile)
            fields = next(reader)
            for row in reader:
                W = np.zeros((3,27))
                scaleFactor = 1
                nameOfAttractor = row[0].lower()
                for index in range(3):
                    for i in range(0,27):
                        element = row[1+(index*27)+i] #+1 to skip the name
                        val = 0
                        if '/' in element:
                            tmp = element.split('/')
                            val = float(tmp[0])/float(tmp[1])
                        else:
                            val=float(element)        
                        W[index,i] = val
                if(self.fitToUnitCube):
                    W,scaleFactor = self.fitWToCube(W)
                tmpDict[nameOfAttractor] = [W,scaleFactor]
            csvFile.close() 
        return tmpDict
    strangeAttractors = {}
    def supports_unicode(self):
    # Check if the stdout encoding is a UTF variant
        encoding = sys.stdout.encoding
        if encoding and 'utf' in encoding.lower():
            return True
        return False
    
    def loadFromCSVString(strArr,fitToCube=doDefaultFitToCube):
        name = strArr.pop(0)
        vec = []
        for element in strArr:
                val = 0
                if '/' in element:
                        tmp = element.split('/')
                        val = float(tmp[0])/float(tmp[1])
                else:
                    val=float(element)
                vec.append(val)   
        WArr= np.reshape(np.array(vec), (3, 27))
        return ODE(W=WArr,doFitToCube=fitToCube),name
    def getCSVString(self):
        tmp = self.W.flatten()
        return (",".join(tmp.astype(str)))
        
    def __init__(self,W = np.zeros((3,27)),doFitToCube = doDefaultFitToCube, name=None):
        self.fitToUnitCube = doFitToCube
        self.strangeAttractors = self.loadStrangeAttractors()
        if name!=None:
            self.W = self.strangeAttractors[name.lower()][0]
            self.scaleFactor = self.strangeAttractors[name.lower()][1]
        else:
            if doFitToCube:
                self.W,self.scaleFactor = self.fitWToCube(W)
            else:
                self.W = W #must be 3x27
                self.scaleFactor = 1.0
        
    def getCorrectSuperscript(self,n,doLatex=False):
        doUnicode = self.supports_unicode()
        if doLatex:
            return f'^{{{n}}}'
        elif doUnicode:
            if n==0:
                return '\u2070'
            elif n==1:
                return '\u00b9'
            elif n==2:
                return '\u00b2'
            else:
                return '\u00b3'
        else:
            return f'^{n}'
    def formatEq(self,coeffs,formatWithLatex=False):
        terms = []
        for i in range(27):
            coeff = coeffs[i]
            if coeff == 0:
                continue
            coeff = round(coeff,3)
            ##Get turnary digits
            d1 = i // 9
            d2 = (i // 3) % 3
            d3 = i % 3
            varPart = ''
            for var, exp in zip(['x','y','z'], [d1, d2, d3]):
                if exp == 1:
                    varPart += var
                elif exp > 1:
                    varPart += f'{var}{self.getCorrectSuperscript(exp,doLatex=formatWithLatex)}'
            if varPart =='':
                varPart = '1'
            if formatWithLatex:
                terms.append(f'{coeff}{varPart}')    
            else:
                terms.append(f'{coeff}*{varPart}')
        if not terms:
            return '0'
        result = terms[0]
        for t in terms[1:]:
            result += f' - {t[1:]}' if t.startswith('-') else f' + {t}'
        return result  
    def printODE(self,doLatex=False):
        ##
        #The poisitions are formed by counting in turnary, with the powers of x y and z being d1,d2, and d3 respectivly
        ##EX: (210)_3=(21)_10=> W[_,21] is the coeff for the x^2*y^1*z^0=x^2*y term
        ##
        W = self.W
        dxTerm = W[0,:]
        dyTerm = W[1,:]
        dzTerm = W[2,:]
        print(f'dx/dt = {self.formatEq(dxTerm,formatWithLatex=doLatex)}')
        print(f'dy/dt = {self.formatEq(dyTerm,formatWithLatex=doLatex)}')
        print(f'dz/dt = {self.formatEq(dzTerm,formatWithLatex=doLatex)}')
    def getLatexFormula(self):
        W = self.W
        dxTerm = W[0,:]
        dyTerm = W[1,:]
        dzTerm = W[2,:]
        formattedDx = f"\\frac{{dx}}{{dt}}&={self.formatEq(dxTerm,formatWithLatex=True)}\\newline"
        formattedDy = f"\\frac{{dy}}{{dt}}&={self.formatEq(dyTerm,formatWithLatex=True)}\\newline"
        formattedDz = f"\\frac{{dz}}{{dt}}&={self.formatEq(dzTerm,formatWithLatex=True)}\\newline"
        return f"\\begin{{align}}{formattedDx}{formattedDy}{formattedDz}\\end{{align}}"
    
    def eval(self, t, pos):
        # Evaluates the ODE at a given time and 3D position. Called on every solver step (not
        # just stiff-mode Jacobian evaluations), so this is the hottest function in the pipeline.
        return self.W @ np.asarray(_monomialFunc(*pos), dtype=np.float64).flatten()

    def computeJacobian(self, pos):
        # J[:,b] = d(eval)/d(var_b). monomialDerivs is (27,3), independent of W - only the
        # basis's own derivatives, symbolically precomputed once at import time (see
        # _monomialJacobianFunc above) and evaluated here via compiled lambdify code.
        monomialDerivs = np.asarray(_monomialJacobianFunc(*pos), dtype=np.float64)
        return self.W @ monomialDerivs

    def computeHessian(self, pos):
        # H[i,b,c] = d^2(eval_i)/d(var_b)d(var_c) - one Hessian matrix per output component
        # (dx/dt, dy/dt, dz/dt), stacked into a rank-3 tensor. monomialHess is (27,3,3),
        # independent of W for the same reason computeJacobian's monomialDerivs is - only the
        # basis changes. einsum generalizes computeJacobian's W @ monomialDerivs to the extra
        # (b,c) axis pair a second derivative introduces, which plain matrix multiply can't.
        monomialHess = np.asarray(_monomialHessianFunc(*pos), dtype=np.float64)
        return np.einsum('ik,kbc->ibc', self.W, monomialHess)

    def jacobianForSolveIvp(self, t, pos):
        # Adapts computeJacobian's (pos-only) signature to solve_ivp's expected jac(t, y).
        return self.computeJacobian(pos)
    
    def integrate(self, initial_pos=None, endTime=maxIntegrationTime, dt=0.005,bound=500,suppressPrint =True,log:logDetails=None,useAnalyticJacobian=False,rtol=1e-3,atol=1e-6,denseOutput=False):
        if initial_pos is None:
            initial_pos = [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]
        if(self.capIntegrationTime and endTime>self.maxIntegrationTime):
            print(f"Integration time exeeds maximum of {round(self.maxIntegrationTime,2)}. It will be capped.")
            endTime = self.maxIntegrationTime
        if(self.fitToUnitCube):#scaling the coeffecents also scales the speed, so account for that
            if (self.capScaleFactorIncrease and (1/self.scaleFactor>=self.maxScaleFactorIncrease)):
                endTime*=self.maxScaleFactorIncrease
            else:
                endTime*=1/self.scaleFactor
            if not suppressPrint:
                print(f"Scaled end time is {round(endTime,2)} using scaling factor of {round(self.scaleFactor,3)}.")
            if(self.capIntegrationTime and endTime>self.maxIntegrationTime):
                if not suppressPrint:
                    print(f"Integration time would exeed maximum of {round(self.maxIntegrationTime,2)}. It will be capped.")
                endTime=self.maxIntegrationTime
        t_eval  = np.arange(0.0, endTime, dt)
        # solve_ivp event: value crosses zero (positive to negative) once any coordinate's
        # magnitude exceeds bound, which halts integration early via terminal=True.
        def blowup(t, y):
            return bound - np.max(np.abs(y))
        blowup.terminal = True
        blowup.direction = -1
        try:
            sol = solve_ivp(
                fun=self.eval,
                t_span=(0,endTime),
                y0=initial_pos,
                method='LSODA',
                t_eval=t_eval,
                events=blowup,
                max_step=dt * 10,
                jac=self.jacobianForSolveIvp if useAnalyticJacobian else None,
                rtol=rtol,
                atol=atol,
                dense_output=denseOutput,
            )
        except Exception:
            if not suppressPrint:
                print("Error with solving IVP")
            if log!=None:
                log.status = "solve_ivp failed"
                log.minDist = "N/A"
            return None
        # Reject: blowup, solver failure, too short, or NaN
        if sol.status == 1:
            if not suppressPrint:
                print("Error:blowup")
            if log!=None:
                log.status = "Blowup"
                log.minDist = "N/A"
            return None
        if not sol.success or len(sol.t) < 20:
            if not suppressPrint:
                print(sol.message)
            if log!=None:
                log.status = "solve_ivp failed"
                log.minDist = "N/A"
            return None
        if not np.all(np.isfinite(sol.y)):
            if not suppressPrint:
                print(sol.message)
            if log!=None:
                log.status = "Infinity in Result"
                log.minDist = "N/A"
            return None
        lastPoint = [sol.y[0][-1],sol.y[1][-1],sol.y[2][-1]]
        secondLastPoint = [sol.y[0][-2],sol.y[1][-2],sol.y[2][-2]]
        # If any two consecutive points are within tol, the trajectory has settled onto a
        # fixed point rather than exhibiting chaotic/periodic motion; flag it as "Stablized".
        tol = 1e-8
        distArr = np.zeros(len(sol.y[0])-1)
        for i in range(len(distArr)):
            lastPoint = [sol.y[0][i],sol.y[1][i],sol.y[2][i]]
            secondLastPoint = [sol.y[0][i+1],sol.y[1][i+1],sol.y[2][i+1]]
            distance = dist.euclidean(lastPoint,secondLastPoint)
            distArr[i]=(distance)
        minDist = min(distArr)
        if (minDist<tol):
            if not suppressPrint:
                print("Stablized")
            if log!=None:
                log.status = "Stablized"
                log.minDist = minDist 
        elif log!=None:
            log.status="Success"
            log.minDist = minDist
        return [sol.t,sol.y,sol.sol] if denseOutput else [sol.t,sol.y]
    def toJitcodeSystem(self):
        dx = sum(self.W[0, i] * self.jcMonomials[i] for i in range(self.W.shape[1]))
        dy = sum(self.W[1, i] * self.jcMonomials[i] for i in range(self.W.shape[1]))
        dz = sum(self.W[2, i] * self.jcMonomials[i] for i in range(self.W.shape[1]))
        f = [dx,dy,dz]
        return f

    def computeLargestLyapunovExponent(self, initial_pos=None, endTime=1000.0,
                                        renormTime=1.0, transientTime=50.0,bound=500,
                                        logStatus:logDetails = None, maxICRetries=4):
        # A single random IC can land somewhere that blows up before settling onto the
        # attractor, even for W that's genuinely chaotic from most starting points (measured:
        # ~60% failure rate on one real candidate). Both JitCODE and the manual fallback
        # previously gave up silently on the first bad IC. Only retry with a fresh random IC
        # when we drew it ourselves - a caller-supplied initial_pos is respected as-is, tried
        # once, not silently swapped out.
        autoGenerateIC = initial_pos is None

        # y is always the 3-element (x,y,z) position here - plain Python max/abs avoids numpy's
        # per-call array dispatch overhead on such a small array. Same check, same values.
        def blowup(y):
            return max(abs(y[0]), abs(y[1]), abs(y[2])) > bound

        # Reused across every retry via set_initial_value (which jitcode_lyap overrides to also
        # draw a fresh random tangent-vector direction each call, so reuse doesn't leak state
        # between attempts). If this fails, it'll fail identically on every attempt regardless
        # of pos, so every attempt below just falls straight through to the manual fallback.
        tmp = None
        try:
            tmp = _newLyapIntegrator()
            tmp.set_parameters(*self.W.flatten().tolist())
        except Exception:
            tmp = None

        for attempt in range(maxICRetries + 1):
            pos = np.array(initial_pos if initial_pos is not None else
                            [random.uniform(-1, 1) for _ in range(3)], dtype=np.float64)
            try:
                if tmp is None:
                    raise Exception("JitCODE integrator unavailable")
                tmp.set_initial_value(pos)
                lyaps = []
                numTransientSteps = max(int(round(transientTime / renormTime)), 0)
                numSteps = max(int(round(endTime / renormTime)), 1)
                times = np.arange(1, numTransientSteps + numSteps + 1) * renormTime
                for time in times:
                    res = tmp.integrate(time)
                    tpos = res[0]
                    if blowup(tpos):
                        raise Exception("Trajectory exceeded bound")
                    lyaps.append(res[1])
                # converting to NumPy array for easier handling
                lyaps = np.vstack(lyaps)
                lyap = np.average(lyaps[numTransientSteps:,0])
                if not(logStatus == None):
                    logStatus.status = "JitCODE Success"
                return lyap
            except Exception as e:
                #Failed, try old method
                manualResult = self.computeLargestLyapunovExponentManual(
                    initial_pos=pos,endTime=endTime,transientTime=transientTime,
                    logStatus=logStatus, renormTime=renormTime)
                if manualResult is not None:
                    return manualResult
                if not autoGenerateIC:
                    break
        if logStatus is not None and logStatus.status == "None":
            logStatus.status = f"Blowup on all {maxICRetries + 1} attempted initial conditions"
        return None

    def computeChaosSignals(self, initial_pos=None, endTime=1000.0, renormTime=1.0,
                             transientTime=50.0, bound=500, logStatus:logDetails = None,
                             maxICRetries=4, corrDimSample=1500):
        # One merged draw: LLE and correlation dimension together, from a single jitcode_lyap
        # integration - the same integration that already produces the LLE also yields the state
        # trajectory needed for correlation dimension, so no second (scipy-based) integration is
        # needed. This roughly halves the cost per draw versus computing them separately, which
        # matters at production scale (measured: real search candidates average ~4 draws each
        # across thousands per round). Retry-on-blowup logic mirrors
        # computeLargestLyapunovExponent; no manual fallback here since the manual integrator
        # doesn't produce a directly comparable trajectory - jitcode-unavailable is rare enough
        # that returning (None, None) for this one draw is acceptable, since callers combine
        # multiple draws and treat a missing one as just contributing nothing.
        autoGenerateIC = initial_pos is None

        # y is always the 3-element (x,y,z) position here - plain Python max/abs avoids numpy's
        # per-call array dispatch overhead on such a small array. Same check, same values.
        def blowup(y):
            return max(abs(y[0]), abs(y[1]), abs(y[2])) > bound

        tmp = None
        try:
            tmp = _newLyapIntegrator()
            tmp.set_parameters(*self.W.flatten().tolist())
        except Exception:
            tmp = None

        numTransientSteps = max(int(round(transientTime / renormTime)), 0)
        numSteps = max(int(round(endTime / renormTime)), 1)
        times = np.arange(1, numTransientSteps + numSteps + 1) * renormTime

        for attempt in range(maxICRetries + 1):
            pos = np.array(initial_pos if initial_pos is not None else
                            [random.uniform(-1, 1) for _ in range(3)], dtype=np.float64)
            try:
                if tmp is None:
                    raise Exception("JitCODE integrator unavailable")
                tmp.set_initial_value(pos)
                lyaps, traj = [], []
                for time in times:
                    res = tmp.integrate(time)
                    tpos = res[0]
                    if blowup(tpos):
                        raise Exception("Trajectory exceeded bound")
                    lyaps.append(res[1])
                    traj.append(tpos.copy())
                lyaps = np.vstack(lyaps)
                lyap = np.average(lyaps[numTransientSteps:, 0])
                traj = np.array(traj[numTransientSteps:])
                corrDim = correlationDimension(traj, nSample=corrDimSample)
                if not (logStatus == None):
                    logStatus.status = "JitCODE Success"
                return lyap, corrDim
            except Exception:
                if not autoGenerateIC:
                    break
        if logStatus is not None and logStatus.status == "None":
            logStatus.status = f"Blowup on all {maxICRetries + 1} attempted initial conditions"
        return None, None

    def computeLargestLyapunovExponentManual(self, initial_pos=None, endTime=1000.0, dt=0.01,
                                        renormTime=1.0, transientTime=50.0, bound=500,
                                        maxMemoryMB=None,logStatus:logDetails = None):
        # Benettin's algorithm: integrate the trajectory alongside a tangent vector v
        # governed by the variational equation dv/dt = J(x(t)) @ v, periodically
        # rescaling v to unit length and averaging the log of its growth factor.
        maxMemoryBytes = (maxMemoryMB if maxMemoryMB is not None else self.maxLyapunovMemoryMB) * 1024 * 1024
        process = psutil.Process()#Make sure we don't overload on memory, as this code experenced memory issues before in results analysis 
        
        # Measured relative to this call's own starting memory, not the process's absolute
        # RSS, since this runs inside larger batch/sweep scripts where other unrelated data
        # may already account for a large, unpredictable baseline.
        baselineRss = process.memory_info().rss
        def memoryLimitExceeded():
            return (process.memory_info().rss - baselineRss) > maxMemoryBytes
        pos = initial_pos
        # solve_ivp events need (t, y) and a zero-crossing value, not a boolean. Only
        # checks y[:3] since v (tangent vector, appended below) can legitimately exceed
        # bound mid-window.
        def blowup(t, y):
            return bound - np.max(np.abs(y[:3]))
        blowup.terminal = True
        blowup.direction = -1
        try:
            # Run the trajectory forward first so it settles onto the attractor before we
            # start measuring; otherwise the initial transient would bias the exponent.
            if transientTime > 0:
                # Only the final point is used below, so restrict solve_ivp's stored output to
                # just that point via t_eval; without it, sol.t/sol.y grow with however many
                # internal steps LSODA needs, which varies a lot across the different ODEs this
                # runs on and can otherwise balloon memory on stiff/near-blowup systems.
                transientSol = solve_ivp(self.eval, (0, transientTime), pos, method='LSODA',
                                          max_step=dt * 10,
                                          jac=self.jacobianForSolveIvp, events=blowup,
                                          t_eval=[transientTime])
                if not transientSol.success or transientSol.status == 1 or not np.all(np.isfinite(transientSol.y)):
                    if not(logStatus==None):
                        logStatus.status = "Blowup (manual transient)"
                    return None
                pos = transientSol.y[:, -1]
                del transientSol
                if memoryLimitExceeded():
                    if not(logStatus==None):
                        logStatus.status = "Memory Error"
                    return None

            # Augmented 6D system: the first 3 components evolve under the ODE as usual, the
            # last 3 (v) evolve under the linearized dynamics dv/dt = J(pos) @ v, tracking how
            # an infinitesimal perturbation at pos grows/shrinks.
            def augmentedEval(t, state):
                p = state[:3]
                v = state[3:]
                return np.concatenate([self.eval(t, p), self.computeJacobian(p) @ v])

            v = np.array([1.0, 0.0, 0.0])  # arbitrary unit-length initial perturbation direction
            numSteps = max(int(round(endTime / renormTime)), 1)
            logSum = 0.0
            for _ in range(numSteps):
                # Integrate pos and v together for one renormTime window.
                state0 = np.concatenate([pos, v])
                # Same reasoning as the transient solve above: only the final point is used,
                # so pin t_eval to it to keep this call's memory footprint independent of how
                # many internal steps the solver takes for this particular ODE/window.
                sol = solve_ivp(augmentedEval, (0, renormTime), state0, method='LSODA',
                                max_step=dt * 10, events=blowup,
                                t_eval=[renormTime])
                if not sol.success or sol.status == 1 or not np.all(np.isfinite(sol.y)):
                    return None
                pos = sol.y[:3, -1]
                v = sol.y[3:, -1]
                del sol
                # v's growth factor over this window estimates the local expansion rate;
                # log it, then rescale v back to unit length so it doesn't over/underflow
                # and stays aligned with the fastest-growing (most unstable) direction.
                vNorm = np.linalg.norm(v)
                if vNorm == 0 or not np.isfinite(vNorm):
                    return None
                logSum += np.log(vNorm)
                v = v / vNorm
                # Checked once per renormTime window (not every solver substep) since
                # process.memory_info() is a syscall and windows are typically short.
                if memoryLimitExceeded():
                    return None

            # The largest Lyapunov exponent is the average exponential growth rate of v,
            # i.e. total log-growth divided by total elapsed time.
            totalTime = numSteps * renormTime
            if not(logStatus==None):
                logStatus.status = "Manual Method Success"
            return logSum / totalTime
        except Exception:
            # Covers unexpected solve_ivp failures (e.g. a MemoryError from a numpy
            # allocation) that the checks above wouldn't otherwise catch, so a pathological
            # ODE aborts cleanly instead of crashing the caller.
            if not(logStatus==None):
                logStatus.status = "Both Methods Failed"
            return None


    def plotInteractive(self,resultOfSolve=None, filename=None,doSave = True,overwrittenEndTime=None,save_to_json = False):
        if resultOfSolve is None:
            try:
                randPos = [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]#random point between 0 and 1
                #print(f"Plotting from {round(randPos[0],2),round(randPos[1],2),round(randPos[2],2)}")
                if overwrittenEndTime!=None:
                    resultOfSolve=self.integrate(initial_pos=randPos,endTime=overwrittenEndTime)
                else:
                    resultOfSolve = self.integrate(initial_pos=randPos)
            except Exception:
                #print("Integration Failed:No solution to plot.")
                return
        if resultOfSolve is None:
            return
        t, y = resultOfSolve
        x_vals, y_vals, z_vals = y[0], y[1], y[2]
        numTicks = 6
        tickvals = np.linspace(t[0], t[-1], numTicks)
        trace = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
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
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        if doSave:
            if filename is None:
                filename = 'attractor'
            if not save_to_json:
                out_path = os.path.join('Plots', f'{filename}.html')
                fig.write_html(out_path)
                #print(f"Saved interactive plot to {out_path}")
            else:
                out_path = os.path.join('JSONsOfPlots', f'{filename}.json')
                json = pio.to_json(fig)
                with open(out_path,mode="w") as f:
                    f.write(json)
                    f.close()
        return fig
    def plotVectorField(self,doSave=True,filename=None):
        def tmpEval(pos):
            return self.eval(0,pos)
        # 2. Generate the 3D coordinate grid ranges
        x_range = np.linspace(-1, 1, 12)
        y_range = np.linspace(-1, 1, 12)
        z_range = np.linspace(-1, 1, 12)

        X, Y, Z = np.meshgrid(x_range, y_range, z_range)

        # 3. Stack the grid arrays into a 2D array of coordinate rows: [[x1, y1, z1], [x2, y2, z2], ...]
        grid_vectors = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        # 4. Apply your function along the vector axis (axis=1 loops over each [x, y, z] row)
        evaluated_vectors = np.apply_along_axis(tmpEval, axis=1, arr=grid_vectors)

        # 5. Extract the separate U, V, W components from the results
        U = evaluated_vectors[:, 0]
        V = evaluated_vectors[:, 1]
        W = evaluated_vectors[:, 2]

        # 6. Build the Plotly figure using the flat coordinate columns
        fig = go.Figure(data=go.Cone(
            x=grid_vectors[:, 0],
            y=grid_vectors[:, 1],
            z=grid_vectors[:, 2],
            u=U,
            v=V,
            w=W,
            colorscale='Cividis',
            sizemode='scaled',
            sizeref=1.8,
            showscale=True,
            colorbar=dict(title="Velocity Magnitude")
        ))

        # 7. Customize layout
        fig.update_layout(
            title="3D Vector Field Plot (Using Vector Inputs)",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                aspectmode='cube'
            ),
            width=900,
            height=800
        )
        if doSave and (filename!=None):
            fig.write_html(f"Plots/{filename}VectorField.html")
        return fig
    def plotMultipleTrajectories(self, initial_positions=None, num_random=5, overwrittenEndTime=None, dt=0.005, filename=None,doSave = True,save_to_json=False):
        # Each trajectory gets a distinct hue, ramping from a dark shade to a bright shade
        # along the path (early points dark, later points bright). Using a per-hue colorscale
        # rather than a single flat color so time-ordering is visible within each trajectory.
        GRADIENT_COLORSCALES = [
            [[0, '#00008B'], [1, '#00BFFF']],  # dark blue → sky blue
            [[0, '#8B0000'], [1, '#FF6666']],  # dark red → bright red
            [[0, '#006400'], [1, '#00FF7F']],  # dark green → spring green
            [[0, '#4B0082'], [1, '#DA70D6']],  # indigo → orchid
            [[0, '#7B3800'], [1, '#FFA500']],  # dark orange → orange
            [[0, '#004444'], [1, '#00FFCC']],  # dark teal → aquamarine
            [[0, '#7B0051'], [1, '#FF69B4']],  # dark pink → hot pink
            [[0, '#5C5C00'], [1, '#FFFF00']],  # dark yellow → yellow
        ]
        if num_random>len(GRADIENT_COLORSCALES):
            num_random=len(GRADIENT_COLORSCALES)
        if initial_positions is None:
            initial_positions = [
                [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]#random point between 0 and 1
                for _ in range(num_random)
            ]
        traces = []
        for i, pos in enumerate(initial_positions):
            if overwrittenEndTime!=None:
                result = self.integrate(pos, endTime=overwrittenEndTime, dt=dt)
            else:
                result = self.integrate(pos, dt=dt)
            if result is None:
                #print(f"Integration failed for {[round(pos[0],2),round(pos[1],2),round(pos[2],2)]}, skipping.")
                continue

            t, y = result
            label = f"({round(pos[0],2)}, {round(pos[1],2)}, {round(pos[2],2)})"
            n = len(y[0])
            # Normalize point index 0→1 so Plotly maps it across the colorscale
            color_vals = [j / max(n - 1, 1) for j in range(n)]
            traces.append(go.Scatter3d(
                x=y[0], y=y[1], z=y[2],
                mode='lines',
                name=label,
                line=dict(
                    width=2,
                    color=color_vals,
                    colorscale=GRADIENT_COLORSCALES[i % len(GRADIENT_COLORSCALES)],
                ),
            ))


        if not traces:
            print("No valid trajectories to plot.")
            return

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(title='Start Position'),
        )

        if doSave:
            if filename is None:
                filename = 'attractor_multi'
            if not save_to_json:
                out_path = os.path.join('Plots', f'{filename}.html')
                fig.write_html(out_path)
                #print(f"Saved interactive plot to {out_path}")
            else:
                out_path = os.path.join('JSONsOfPlots', f'{filename}.json')
                json = pio.to_json(fig)
                with open(out_path,mode="w") as f:
                    f.write(json)
                    f.close()
        return fig
    # Static (matplotlib) counterpart to plotInteractive's time-colored Plotly trace -
    # for print/PDF figures rather than browser-viewed HTML. Draws onto a caller-supplied
    # Axes3D (ax) when composing a multi-panel grid, or creates its own figure otherwise.
    # x/y/z need not come from self.integrate() - e.g. a saved discovery trajectory's own
    # x/y/z arrays work directly - this method only draws, it doesn't integrate.
    #
    # Default colormap is matplotlib's own built-in 'plasma' - chosen to visually match
    # plotInteractive's colorscale="Plasma" above, so this project's static (print/PDF)
    # and interactive (Plotly HTML) trajectory renders read as the same visual language.
    # This doesn't pull in Plotly or its own Plasma implementation - "plasma" is a
    # perceptually-uniform colormap matplotlib itself shipped first (matplotlib 2.0;
    # Plotly and others later adopted it under the same name), so matching it here is
    # just matplotlib's own builtin, nothing imported from plotly.
    def plotStaticGradient(self, x, y, z, ax=None, elev=20, azim=-60, burnInFrac=0.1,
                            linewidth=0.5, cmap='plasma', title=None):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        burn = int(len(x) * burnInFrac)
        x, y, z = x[burn:], y[burn:], z[burn:]
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Normalized position along the (already burned-in) trajectory stands in for time:
        # light = earliest point plotted, dark = latest.
        tNorm = np.linspace(0, 1, len(segments))

        ownFig = ax is None
        if ownFig:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(projection='3d')

        lc = Line3DCollection(segments, cmap=cmap, array=tNorm, linewidth=linewidth)
        ax.add_collection3d(lc)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        # Orthographic (not matplotlib's default perspective) projection, so a part of the
        # trajectory farther from the camera isn't rendered smaller than a part that's
        # closer - true relative proportions regardless of depth, which matters more here
        # than the photographic depth cues perspective gives. zoom>1 on an equal (default)
        # box_aspect shrinks mplot3d's considerable default margin so the trajectory fills
        # more of the panel, while every panel in a multi-panel figure still gets the same
        # size box (aspect itself isn't scaled per-trajectory).
        ax.set_proj_type('ortho')
        ax.set_box_aspect(None, zoom=1.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if title:
            ax.set_title(title, fontsize=10)

        if ownFig:
            return fig, ax, lc
        return ax, lc

    def saveTrajectoryPlots(self,filenameToSave,saveJson=False):
        self.plotInteractive(filename=filenameToSave)
        self.plotMultipleTrajectories(filename=(f'{filenameToSave}Multi'))
        if saveJson:
            self.plotInteractive(filename=filenameToSave,save_to_json=True)
            self.plotMultipleTrajectories(filename=(f'{filenameToSave}Multi'),save_to_json=True)