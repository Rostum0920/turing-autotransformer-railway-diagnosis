#!/usr/bin/env python3
"""Baseline solver of autotransformer-railway-diagnosis (an author-side measurement, not the reference): the exact coupled model of the instruction with greedy forward selection (the best single addition that gains more than 40, with exact refits) followed by backward elimination; no other move. Readings beyond 6 standard deviations are the gross errors. Every search budget is a count and not a time."""
import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
FLAG = 6.0
ADD = 40.0

HUBER = 2.0
CLASSES = ("cat", "fdr", "rail", "leak", "at", "off", "gain")
BOUND = {"cat": (0.0, 0.88), "fdr": (0.0, 1.1), "rail": (0.0, 1.76), "leak": (0.0, 4.4), "off": (-88.0, 88.0), "gain": (-0.132, 0.132)}
STEP = {"cat": 1e-4, "fdr": 1e-4, "rail": 1e-4, "leak": 1e-4, "off": 1e-2, "gain": 1e-5}                                                                # finite-difference steps
TRIAL = {"cat": (0.2, 0.5, 0.8), "fdr": (0.25, 0.6, 1.0), "rail": (0.4, 1.0, 1.6), "leak": (1.0, 2.5, 4.0), "at": (1.0,), "off": (-80.0, -50.0, -20.0, 20.0, 50.0, 80.0), "gain": (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12)}


class Instance:
    """railway line + operating points + sensors (+ readings), built from the specification.  Every section has the 3 x 3 series impedance matrix of its contact wire, rail and negative feeder (self and mutual terms).
    For a configuration of errors the network solve is a low-rank (Woodbury) update of the solve for the nominal network."""
    def __init__(self, spec, z=None):
        self.spec = spec; self.L = np.array(spec["lengths"]); self.M = len(self.L) + 1; M = self.M; S = M - 1; N3 = 3 * M; ln = spec["constants"]
        self.subs = [s["node"] for s in spec["substations"]]; self.zs = np.array([s["r"] + 1j * s["x"] for s in spec["substations"]]); self.ats = list(spec["autotransformers"])
        zc = ln["r_contact"] + 1j * ln["x_contact"]; zr = ln["r_rail"] + 1j * ln["x_rail"]; zf = ln["r_feeder"] + 1j * ln["x_feeder"]
        zcr = ln["r_mutual"] + 1j * ln["x_mutual_cr"]; zcf = ln["r_mutual"] + 1j * ln["x_mutual_cf"]; zrf = ln["r_mutual"] + 1j * ln["x_mutual_rf"]
        self.Zk = np.array([[zc, zcr, zcf], [zcr, zr, zrf], [zcf, zrf, zf]])                                       # per kilometre; conductor order: contact wire 0, rail 1, negative feeder 2
        self.selfz = np.stack([self.L * zc, self.L * zr, self.L * zf], axis=1)                                        # (S, 3) self impedance of every conductor of every section
        self.Y0 = np.linalg.inv(self.L[:, None, None] * self.Zk[None])                                               # (S, 3, 3) admittance matrix of the section: currents = Y0 @ voltage drops
        self.g0 = np.zeros(M); self.g0[:-1] += 0.5 * ln["g_leak"] * self.L; self.g0[1:] += 0.5 * ln["g_leak"] * self.L; self.rg = ln["r_ground"]; self.zat = ln["r_at"] + 1j * ln["x_at"]
        self.gh = 0.5 * ln["g_leak"] * self.L                                                                       # leakage conductance of a section shared to each of its end nodes
        ops = spec["operating_points"]; self.K = len(ops); self.sensors = spec["sensors"]; self.nm = len(self.sensors); self.sig = np.array([s["sigma"] for s in self.sensors]); self.z = z; self._cache = {}
        self.J = np.zeros((self.K, N3), complex)
        for k, op in enumerate(ops):
            for t in op["trains"]:
                ph = t["current"] * np.exp(-1j * np.deg2rad(t["angle_deg"])); self.J[k, t["node"]] -= ph; self.J[k, M + t["node"]] += ph
        self.vsens = [s["index"] for s in self.sensors if s["type"] in ("contact_voltage", "feeder_voltage")]; self.msens = [s["index"] for s in self.sensors if "current" in s["type"]]
        self.un = 27500.0; self.R = len(self.subs) + 1; self.yat = 1.0 / (2 * self.zat)
        G0 = np.zeros((N3, N3), complex)
        Bm = np.zeros((3, 6)); Bm[0, 0] = Bm[1, 2] = Bm[2, 4] = 1.0; Bm[0, 1] = Bm[1, 3] = Bm[2, 5] = -1.0                  # drops of the three conductors from the six end voltages of a section
        self._sidx = [np.array([j, j + 1, M + j, M + j + 1, 2 * M + j, 2 * M + j + 1]) for j in range(S)]; self._Bm = Bm
        for j in range(S): G0[np.ix_(self._sidx[j], self._sidx[j])] += Bm.T @ self.Y0[j] @ Bm
        G0[M + np.arange(M), M + np.arange(M)] += self.g0
        e = np.zeros(N3, complex)
        for q, n in enumerate(self.subs):
            ys = 1.0 / self.zs[q]; C, R, F = n, M + n, 2 * M + n
            G0[C, C] += ys; G0[R, R] += ys; G0[C, R] -= ys; G0[R, C] -= ys; e[C] += ys; e[R] -= ys                    # winding 1: rail (-) to contact wire (+)
            G0[R, R] += ys; G0[F, F] += ys; G0[R, F] -= ys; G0[F, R] -= ys; e[R] += ys; e[F] -= ys                    # winding 2: negative feeder (-) to rail (+)
            G0[R, R] += 1.0 / self.rg                                                                                  # rail earthing at the substation
        d = np.array([1.0, 1.0, -2.0])
        for n in self.ats:
            idx = np.array([n, 2 * M + n, M + n]); G0[np.ix_(idx, idx)] += self.yat * np.outer(d, d)
        self.X0 = np.linalg.solve(G0, np.concatenate([e[:, None], self.J.T], axis=1))
        cols = []; self.upd_index = {}
        for cl, ci in (("cat", 0), ("rail", 1), ("fdr", 2)):
            for j in range(S):                                                                                          # an error of the self impedance of one conductor of a section: column B^T Y0 e_i
                u = np.zeros(N3, complex); u[self._sidx[j]] = Bm.T @ self.Y0[j][:, ci]; self.upd_index[(cl, j)] = [len(cols)]; cols.append(u)
        for j in range(S):                                                                                              # a rail-leakage error adds to the rail-to-earth conductance at both end nodes of the section
            c2 = []
            for nd in (j, j + 1):
                u = np.zeros(N3, complex); u[M + nd] = 1; c2.append(len(cols)); cols.append(u)
            self.upd_index[("leak", j)] = c2
        for n in self.ats:
            u = np.zeros(N3, complex); u[n] = 1; u[2 * M + n] = 1; u[M + n] = -2; self.upd_index[("at", n)] = [len(cols)]; cols.append(u)
        U = np.array(cols).T; self.W = np.linalg.solve(G0, U); self.Gram = U.T @ self.W; self.UX = U.T @ self.X0
        nm = self.nm; self.a_idx = np.zeros(nm, int); self.b_idx = np.full(nm, N3, int); self.s_kind = np.zeros(nm, int); self.s_sec = np.zeros(nm, int); self.s_cond = np.zeros(nm, int); self.s_const = np.zeros(nm, complex); self.s_fix = np.ones(nm, complex)
        for s in self.sensors:
            i, t = s["index"], s["type"]
            if t == "contact_voltage": n = s["node"]; self.a_idx[i] = n; self.b_idx[i] = M + n
            elif t == "feeder_voltage": n = s["node"]; self.a_idx[i] = M + n; self.b_idx[i] = 2 * M + n
            elif t == "rail_voltage": n = s["node"]; self.a_idx[i] = M + n
            elif t == "substation_current":
                n = s["node"]; ys = 1.0 / self.zs[self.subs.index(n)]; self.a_idx[i] = n; self.b_idx[i] = M + n; self.s_fix[i] = -ys; self.s_const[i] = ys
            else:
                j = s["section"]; self.s_kind[i] = 1; self.s_sec[i] = j
                self.s_cond[i] = {"contact_current": 0, "rail_current": 1, "feeder_current": 2}[t]
        self.cur = np.where(self.s_kind == 1)[0]
        self.c_sec = self.s_sec[self.cur]; self.c_cond = self.s_cond[self.cur]
        self.c_a = np.stack([self.c_sec, M + self.c_sec, 2 * M + self.c_sec], axis=1); self.c_b = self.c_a + 1
        self.Yrow0 = self.Y0[self.c_sec, self.c_cond, :] if len(self.cur) else np.zeros((0, 3), complex)
        self.cur_in_sec = {}
        for p, j in enumerate(self.c_sec): self.cur_in_sec.setdefault(int(j), []).append(p)

    def _owner(self, cl, k):
        """number of substations before the location of an error: the errors with the same owner act between the same two substations"""
        if cl == "at": p = float(k)
        elif cl in ("cat", "fdr", "rail", "leak"): p = k + 0.5
        else:
            sn = self.sensors[k]; p = sn["node"] if "node" in sn else sn["section"] + 0.5
        return sum(1 for n in self.subs if n < p)

    def _net(self, cfg, cache=True):
        key = tuple(sorted((cl, str(k), round(float(v), 10)) for cl in ("cat", "fdr", "rail", "leak", "at") for k, v in cfg[cl].items())) if cache else None
        hit = self._cache.get(key) if cache else None
        if hit is not None: return hit
        idx = []; blocks = []; secy = {}; secs = {}
        for cl, ci in (("cat", 0), ("rail", 1), ("fdr", 2)):
            for j, v in cfg[cl].items(): secs.setdefault(int(j), {})[ci] = v
        name = ("cat", "rail", "fdr")
        for j, dv in secs.items():                                                                                      # all errors of one section: Woodbury on its 3 x 3 admittance matrix
            ii = sorted(dv); Dl = np.diag([dv[i] * self.selfz[j, i] for i in ii]); Kb = self.Y0[j][np.ix_(ii, ii)]
            Mx = np.linalg.solve(np.eye(len(ii)) + Dl @ Kb, Dl)
            blocks.append((len(idx), -Mx)); idx += [self.upd_index[(name[i], j)][0] for i in ii]
            secy[j] = self.Y0[j] - self.Y0[j][:, ii] @ Mx @ self.Y0[j][ii, :]
        for j, v in cfg["leak"].items():
            blocks.append((len(idx), np.diag([v * self.gh[j]] * 2).astype(complex))); idx += self.upd_index[("leak", j)]
        for n in cfg["at"]: blocks.append((len(idx), np.array([[-self.yat]]))); idx += self.upd_index[("at", n)]
        X = self.X0
        if idx:
            n = len(idx); C = np.zeros((n, n), complex)
            for st, blk in blocks: m = blk.shape[0]; C[st:st + m, st:st + m] = blk
            idx = np.array(idx); Gi = self.Gram[np.ix_(idx, idx)]
            X = self.X0 - self.W[:, idx] @ (C @ np.linalg.solve(np.eye(n) + Gi @ C, self.UX[idx]))
        out = (X[:, 1:].T, X[:, 0], secy)
        if cache:
            if len(self._cache) > 50000: self._cache.clear()
            self._cache[key] = out
        return out

    def coeffs(self, cfg, cache=True):
        """complex A (nm,) and B (K, nm) with model reading gain * |U A + B| + offset, U the unknown source voltage of the operating point"""
        V0, V1, secy = self._net(cfg, cache)
        V1x = np.append(V1, 0); V0x = np.concatenate([V0, np.zeros((self.K, 1), complex)], axis=1)
        A = self.s_fix * (V1x[self.a_idx] - V1x[self.b_idx]) + self.s_const; B = self.s_fix[None, :] * (V0x[:, self.a_idx] - V0x[:, self.b_idx])
        if len(self.cur):
            Yr = self.Yrow0
            if secy:
                Yr = Yr.copy()
                for j, Yp in secy.items():
                    for p in self.cur_in_sec.get(j, ()): Yr[p] = Yp[self.c_cond[p]]
            A[self.cur] = (Yr * (V1x[self.c_a] - V1x[self.c_b])).sum(1); B[:, self.cur] = (Yr[None] * (V0x[:, self.c_a] - V0x[:, self.c_b])).sum(2)
        OFF = np.zeros(self.nm); GAIN = np.ones(self.nm)
        for k, v in cfg["off"].items(): OFF[k] = v
        for k, v in cfg["gain"].items(): GAIN[k] = 1 + v
        return A, B, OFF, GAIN
S_CLIP = (0.85, 1.15)


def predict(inst, cfg, N):
    """noise-free readings (K, nm) of a configuration for the nuisances N (K, 2) = (source voltage, train-current scale) of the operating points"""
    A, B, OFF, GAIN = inst.coeffs(cfg); return GAIN * np.abs(N[:, 0:1] * A[None, :] + N[:, 1:2] * B) + OFF


def fit(inst, cfg, z=None, U0=None, iters=6):
    """robust objective and whitened residuals after the fit of the two nuisances of every operating point, the source voltage U and the train-current scale s (Gauss-Newton with Huber weights); the values of cfg are NOT adjusted.
    Returns (J, residuals, N) with N (K, 2) = (U, s)"""
    z = inst.z if z is None else z; A, B, OFF, GAIN = inst.coeffs(cfg); sg = inst.sig[None, :]; h = HUBER * 1.5
    N = np.column_stack([np.full(inst.K, inst.un), np.ones(inst.K)]) if U0 is None else np.array(U0, float).copy()
    for it in range(iters):
        W = N[:, 0:1] * A[None, :] + N[:, 1:2] * B; m = np.abs(W) + 1e-12; res = (z - (GAIN * m + OFF)) / sg; w = np.minimum(1.0, h / np.maximum(np.abs(res), 1e-9))
        dU = GAIN * (W.real * A.real[None, :] + W.imag * A.imag[None, :]) / m / sg; dS = GAIN * (W.real * B.real + W.imag * B.imag) / m / sg
        a11 = (w * dU * dU).sum(1) * (1 + 1e-6) + 1e-12; a22 = (w * dS * dS).sum(1) * (1 + 1e-6) + 1e-12; a12 = (w * dU * dS).sum(1); b1 = (w * dU * res).sum(1); b2 = (w * dS * res).sum(1)
        det = a11 * a22 - a12 * a12 + 1e-30; N[:, 0] += (a22 * b1 - a12 * b2) / det; N[:, 1] += (a11 * b2 - a12 * b1) / det
        N[:, 0] = np.clip(N[:, 0], 15000.0, 40000.0); N[:, 1] = np.clip(N[:, 1], *S_CLIP)
    W = N[:, 0:1] * A[None, :] + N[:, 1:2] * B; m = np.abs(W) + 1e-12; r = (z - (GAIN * m + OFF)) / sg; a = np.abs(r)
    return 2 * np.where(a <= h, a * a / 2, h * a - h * h / 2).sum(), r, N


def empty(): return {cl: {} for cl in CLASSES}


def hyps_of(cfg): return [(cl, k, cfg[cl][k]) for cl in CLASSES for k in sorted(cfg[cl], key=str)]


def clone(cfg): return {cl: dict(cfg[cl]) for cl in CLASSES}


def remove(cfg, h): c = clone(cfg); c[h[0]].pop(h[1]); return c


def apply(cfg, h): c = clone(cfg); c[h[0]][h[1]] = h[2]; return c


def key_of(cfg): return tuple((cl, str(k)) for cl, k, v in hyps_of(cfg))


def domain(T):
    return {"cat": list(range(T.M - 1)), "fdr": list(range(T.M - 1)), "rail": list(range(T.M - 1)), "leak": list(range(T.M - 1)), "at": list(T.ats), "off": list(T.vsens), "gain": list(T.msens)}


def _rh(r, h=HUBER * 1.5):
    a = np.abs(r); return np.sign(r) * np.sqrt(2 * np.where(a <= h, a * a / 2, h * a - h * h / 2))


def refit(inst, cfg, iters=15, U0=None):
    """robust fit of the continuous values of the errors of cfg (Levenberg-Marquardt, finite-difference Jacobian, box bounds) with the source voltages fitted inside every evaluation; returns (J, fitted cfg, U)"""
    keys = [(cl, k) for cl in CLASSES if cl != "at" for k in sorted(cfg[cl], key=str)]
    J0, r0, U = fit(inst, cfg, U0=U0, iters=8)
    if not keys: return J0, clone(cfg), U
    th = np.array([cfg[cl][k] for cl, k in keys], float); lo = np.array([BOUND[cl][0] for cl, k in keys]); hi = np.array([BOUND[cl][1] for cl, k in keys]); st = np.array([STEP[cl] for cl, k in keys])
    def build(t):
        c = clone(cfg)
        for (cl, k), v in zip(keys, t): c[cl][k] = float(v)
        return c
    rh = _rh(r0).ravel(); cost = float(rh @ rh); lam = 1e-2
    for it in range(iters):
        Jm = np.empty((rh.size, len(keys)))
        for i in range(len(keys)):
            t2 = th.copy(); t2[i] += st[i]; Ji, ri, _ = fit(inst, build(t2), U0=U, iters=2); Jm[:, i] = (_rh(ri).ravel() - rh) / st[i]
        g = Jm.T @ rh; H = Jm.T @ Jm; ok = False
        for _ in range(8):
            try: step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
            except np.linalg.LinAlgError: lam *= 10; continue
            t2 = np.clip(th + step, lo, hi); J2, r2, U2 = fit(inst, build(t2), U0=U, iters=4); c2 = float(_rh(r2).ravel() @ _rh(r2).ravel())
            if c2 < cost - 1e-9: ok = True; break
            lam *= 5
        if not ok: break
        gain = cost - c2; th, cost, U, rh = t2, c2, U2, _rh(r2).ravel(); lam = max(lam * 0.3, 1e-7)
        if gain < 0.3: break
    c = build(th); Jf, rf, Uf = fit(inst, c, U0=U, iters=8); return Jf, c, Uf


def _ck(cfg): return tuple((cl, str(k), round(float(v), 7)) for cl in CLASSES for k, v in sorted(cfg[cl].items(), key=lambda t: str(t[0])))


DELTA = {"cat": 0.02, "fdr": 0.02, "rail": 0.04, "leak": 0.1, "off": 2.0, "gain": 0.004}
CLIP = {"cat": (0.1, 0.8), "fdr": (0.15, 1.0), "rail": (0.2, 1.6), "leak": (0.5, 4.0), "off": (-80.0, 80.0), "gain": (-0.12, 0.12)}


def _profile(inst, cfg, N0, r0, dom, allowed, skip, J0):
    """coupling-aware ranking of the single additions: weighted least-squares gain of the new column after projecting out the columns of the current hypotheses"""
    h = HUBER * 1.5; w = np.minimum(1.0, h / np.maximum(np.abs(r0), 1e-9)); sw = np.sqrt(w).ravel(); rw = sw * r0.ravel(); cols = []
    for (cl, k, v) in hyps_of(cfg):
        if cl == "at": continue
        r1 = fit(inst, apply(cfg, (cl, k, v + DELTA[cl])), U0=N0, iters=3)[1]; cols.append(sw * ((r1 - r0) / DELTA[cl]).ravel())
    if cols:
        Q, _ = np.linalg.qr(np.array(cols).T); rt = rw - Q @ (Q.T @ rw)
    else: Q = None; rt = rw
    out = []
    for cl in ("cat", "fdr", "rail", "leak", "off", "gain"):
        if cl not in allowed: continue
        for k in dom[cl]:
            if k in cfg[cl] or (cl, k) in skip: continue
            r1 = fit(inst, apply(cfg, (cl, k, DELTA[cl])), U0=N0, iters=3)[1]; c = sw * ((r1 - r0) / DELTA[cl]).ravel()
            if Q is not None: c = c - Q @ (Q.T @ c)
            den = float(c @ c) + 1e-12; num = float(c @ rt); gain = num * num / den; th = -num / den
            lo, hi = CLIP[cl]
            if cl in ("off", "gain"):
                if abs(th) < 0.5 * abs(lo if cl == "gain" else 20.0) and cl == "gain": th = np.sign(th) * 0.04 if th != 0 else 0.04
                th = float(np.clip(th, lo, hi))
            else: th = float(np.clip(th, lo, hi))
            out.append((gain, (cl, k, th)))
    out.sort(key=lambda t: -t[0]); return out


def top_additions(inst, cfg, n=8, allowed=CLASSES, skip=()):
    """the best single additions (class, location, trial value) by the objective with every other value fixed, plus the two best of each class; skip lists (class, location) pairs not to propose"""
    memo = inst.__dict__.setdefault("_ta", {}); mkey = (_ck(cfg), n, tuple(allowed), tuple(sorted(skip)))
    if mkey in memo: return memo[mkey]
    dom = domain(inst); res = []
    J0, r0, N0 = fit(inst, cfg); sg = inst.sig; mdl = inst.z - r0 * sg[None, :]
    for cl in CLASSES:
        if cl not in allowed: continue
        for k in dom[cl]:
            if k in cfg[cl] or (cl, k) in skip: continue
            trials = list(TRIAL[cl])
            if cl == "off":
                o = float(np.median(r0[:, k]) * sg[k])
                if abs(o) >= 10.0: trials.append(float(np.clip(o, -80.0, 80.0)))
            elif cl == "gain":
                g = float(np.median(r0[:, k] * sg[k] / np.maximum(mdl[:, k], 1e-6)))
                if abs(g) >= 0.02: trials.append(float(np.clip(g, -0.12, 0.12)))
            best = min(((fit(inst, apply(cfg, (cl, k, float(v))), U0=N0, iters=3)[0], (cl, k, float(v))) for v in trials), key=lambda t: t[0]); res.append(best)
    pr = _profile(inst, cfg, N0, r0, dom, allowed, skip, J0); extra = []
    for gain, hh in pr[:n] + [t for cl in allowed for t in [t for t in pr if t[1][0] == cl][:2]]:
        Jf = fit(inst, apply(cfg, hh), U0=N0, iters=3)[0]; extra.append((min(Jf, J0 - 0.8 * gain), hh))
    res.sort(key=lambda t: t[0]); sel = res[:n] + extra
    for cl in allowed: sel += [t for t in res if t[1][0] == cl][:2]
    best_by = {}
    for t in sel:
        key = (t[1][0], t[1][1])
        if key not in best_by or t[0] < best_by[key][0]: best_by[key] = t
    out = sorted(best_by.values(), key=lambda t: t[0])
    if len(memo) > 20000: memo.clear()
    memo[mkey] = out; return out


def pen(c): return ADD * len(hyps_of(c))
def tot(st): return st[0] + pen(st[1])
def mk(inst, cfg, U0=None):
    memo = inst.__dict__.setdefault("_mk", {}); key = _ck(cfg)
    if key not in memo:
        J, c, U = refit(inst, cfg, U0=U0); memo[key] = (J, c, U)
    return memo[key]


def forward(inst, start, allowed=CLASSES, width=3, depth=16):
    best = start; level = [start]
    for _ in range(depth):
        pool = {}
        for st in level:
            for Jn, h in top_additions(inst, st[1], 10, allowed):
                if st[0] - Jn < 0.25 * ADD: continue
                s1 = mk(inst, apply(st[1], h), st[2])
                if st[0] - s1[0] > ADD: pool[key_of(s1[1])] = s1
        if not pool: break
        new = sorted(pool.values(), key=lambda s: s[0]); level = new[:width]
        for cl in allowed:
            for s in new:
                if s not in level and any(h[0] == cl for h in hyps_of(s[1])) and len(level) < width + 3: level.append(s); break
        if tot(level[0]) < tot(best): best = level[0]
    return best


def prune(inst, st):
    again = True
    while again:
        again = False
        for h in hyps_of(st[1]):
            s1 = mk(inst, remove(st[1], h), st[2])
            if tot(s1) < tot(st) - 1e-6: st = s1; again = True; break
    return st


def swap_moves(inst, st, keep=5):
    """replace one hypothesis by another: same-class relocations at its value and the best additions, screened with the values fixed and refitted for the best few"""
    dom = domain(inst); best = st
    for h in hyps_of(st[1]):
        c0 = remove(st[1], h); base = mk(inst, c0, st[2]); own = (h[0], h[1])
        cands = [(h[0], k, h[2]) for k in dom[h[0]] if k not in c0[h[0]] and k != h[1]]
        cands += [hh for _, hh in top_additions(inst, c0, 8, CLASSES, skip={own})]
        seen = set(); sc = []
        for hh in cands:
            if (hh[0], hh[1]) in seen or (hh[0], hh[1]) == own: continue
            seen.add((hh[0], hh[1])); sc.append((fit(inst, apply(c0, hh))[0], hh))
        sc.sort(key=lambda t: t[0])
        for _, hh in sc[:keep]:
            s1 = mk(inst, apply(c0, hh), base[2])
            if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def pair_add(inst, st, top=18, keep=20):
    singles = [h for _, h in top_additions(inst, st[1], top, CLASSES)]; cand = []
    for a in range(len(singles)):
        for b in range(a + 1, len(singles)):
            if singles[a][:2] == singles[b][:2]: continue
            c = apply(apply(st[1], singles[a]), singles[b]); cand.append((fit(inst, c)[0], c))
    cand.sort(key=lambda t: t[0]); best = st
    for _, c in cand[:keep]:
        s1 = mk(inst, c, st[2])
        if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def _best_pairs(inst, c0, singles, own, keep):
    cand = []
    for a in range(len(singles)):
        for b in range(a + 1, len(singles)):
            if singles[a][:2] == singles[b][:2] or singles[a][:2] in own or singles[b][:2] in own: continue
            c = apply(apply(c0, singles[a]), singles[b]); cand.append((fit(inst, c)[0], c))
    cand.sort(key=lambda t: t[0]); return [c for _, c in cand[:keep]]


def _pos(inst, cl, k):
    if cl == "at": return float(k)
    if cl in ("cat", "fdr", "rail", "leak"): return k + 0.5
    sn = inst.sensors[k]; return float(sn["node"]) if "node" in sn else sn["section"] + 0.5


LOC = 4.0


def two_for_two(inst, st, top=8, keep=4):
    hs = hyps_of(st[1]); best = st
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if inst._owner(hs[i][0], hs[i][1]) != inst._owner(hs[j][0], hs[j][1]) or abs(_pos(inst, hs[i][0], hs[i][1]) - _pos(inst, hs[j][0], hs[j][1])) > LOC: continue
            c0 = remove(remove(st[1], hs[i]), hs[j]); own = {hs[i][:2], hs[j][:2]}; base = mk(inst, c0, st[2])
            singles = [h for _, h in top_additions(inst, c0, top, CLASSES, skip=own)]
            for c in _best_pairs(inst, c0, singles, own, keep):
                s1 = mk(inst, c, base[2])
                if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def merge_moves(inst, st, top=8, keep=4):
    """replace two hypotheses acting in the same zone by the best single other hypothesis"""
    hs = hyps_of(st[1]); best = st
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if inst._owner(hs[i][0], hs[i][1]) != inst._owner(hs[j][0], hs[j][1]) or abs(_pos(inst, hs[i][0], hs[i][1]) - _pos(inst, hs[j][0], hs[j][1])) > LOC: continue
            c0 = remove(remove(st[1], hs[i]), hs[j]); own = {hs[i][:2], hs[j][:2]}; base = mk(inst, c0, st[2])
            if tot(base) < tot(best) - 1e-6: best = base
            sc = sorted(((fit(inst, apply(c0, h))[0], h) for _, h in top_additions(inst, c0, top, CLASSES, skip=own)), key=lambda t: t[0])
            for _, h in sc[:keep]:
                s1 = mk(inst, apply(c0, h), base[2])
                if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def split_moves(inst, st, top=8, keep=4):
    """replace one hypothesis by the best two other hypotheses"""
    best = st
    for h in hyps_of(st[1]):
        c0 = remove(st[1], h); own = {h[:2]}; base = mk(inst, c0, st[2])
        singles = [hh for _, hh in top_additions(inst, c0, top, CLASSES, skip=own)]
        for c in _best_pairs(inst, c0, singles, own, keep):
            s1 = mk(inst, c, base[2])
            if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def shift_pairs(inst, st, keep=4):
    """move two network hypotheses of the same zone by up to two sections each, at their values"""
    dom = domain(inst); hs = [h for h in hyps_of(st[1]) if h[0] in ("cat", "fdr", "rail", "leak")]; best = st; S = inst.M - 1
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if inst._owner(hs[i][0], hs[i][1]) != inst._owner(hs[j][0], hs[j][1]) or abs(_pos(inst, hs[i][0], hs[i][1]) - _pos(inst, hs[j][0], hs[j][1])) > LOC: continue
            c0 = remove(remove(st[1], hs[i]), hs[j]); base = mk(inst, c0, st[2]); cand = []
            for da in (-2, -1, 0, 1, 2):
                for db in (-2, -1, 0, 1, 2):
                    if da == 0 and db == 0: continue
                    ka, kb = hs[i][1] + da, hs[j][1] + db
                    if not (0 <= ka < S and 0 <= kb < S) or ka in c0[hs[i][0]] or kb in c0[hs[j][0]] or (hs[i][0] == hs[j][0] and ka == kb): continue
                    c = apply(apply(c0, (hs[i][0], ka, hs[i][2])), (hs[j][0], kb, hs[j][2])); cand.append((fit(inst, c)[0], c))
            cand.sort(key=lambda t: t[0])
            for _, c in cand[:keep]:
                s1 = mk(inst, c, base[2])
                if tot(s1) < tot(best) - 1e-6: best = s1
    return best


def refine(inst, st, rounds=8):
    for _ in range(rounds):
        before = tot(st)
        for move in (prune, swap_moves, pair_add, two_for_two, shift_pairs, merge_moves, split_moves, prune):
            s1 = move(inst, st)
            if tot(s1) < tot(st) - 1e-6: st = s1
        if tot(st) > before - 1e-6: break
    return st


def diagnose(inst):
    """textbook greedy forward selection with exact refits (best single addition while it gains more than ADD), then backward elimination"""
    st = mk(inst, empty())
    while True:
        cands = []
        for Jn, h in top_additions(inst, st[1], 10, CLASSES):
            if st[0] - Jn < 0.25 * ADD: continue
            s1 = mk(inst, apply(st[1], h), st[2])
            if st[0] - s1[0] > ADD: cands.append(s1)
        if not cands: break
        st = min(cands, key=lambda t: t[0])
    return prune(inst, st)


def report(inst, st):
    J, cfg, U = st; r = fit(inst, cfg, U0=U, iters=8)[1]
    out = {"contact_wire_increases": {str(k): round(float(v), 6) for k, v in sorted(cfg["cat"].items())}, "feeder_increases": {str(k): round(float(v), 6) for k, v in sorted(cfg["fdr"].items())},
           "rail_increases": {str(k): round(float(v), 6) for k, v in sorted(cfg["rail"].items())}, "rail_leakage_increases": {str(k): round(float(v), 6) for k, v in sorted(cfg["leak"].items())},
           "autotransformers_out": sorted(int(n) for n in cfg["at"]), "voltage_sensor_offsets": {str(k): round(float(v), 6) for k, v in sorted(cfg["off"].items())},
           "current_meter_gains": {str(k): round(100.0 * float(v), 6) for k, v in sorted(cfg["gain"].items())}}
    out["bad_measurements"] = [[int(a), int(b)] for a, b in np.argwhere(np.abs(r) > FLAG)]
    return out


def solve_line(args):
    indir, lid = args
    spec = json.loads((Path(indir) / f"line_{lid}.json").read_text())
    with np.load(Path(indir) / f"readings_{lid}.npz") as f: z = np.array(f["z"], float)
    inst = Instance(spec, z); return lid, report(inst, diagnose(inst))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input-dir", required=True); ap.add_argument("--output-dir", required=True); a = ap.parse_args()
    ids = [str(x) for x in json.loads((Path(a.input_dir) / "campaign.json").read_text())["lines"]]
    jobs = [(a.input_dir, lid) for lid in ids]
    try:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(2) as pool: res = dict(pool.map(solve_line, jobs, chunksize=1))
    except Exception:
        res = dict(solve_line(j) for j in jobs)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.output_dir) / "diagnosis.json").write_text(json.dumps({"lines": {lid: res[lid] for lid in ids}}, indent=1))


if __name__ == "__main__":
    main()
