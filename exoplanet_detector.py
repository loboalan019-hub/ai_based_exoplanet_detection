"""
Exoplanet Detector — Python port
=================================
A direct translation of the JavaScript logic used in the "Orbital Pioneers"
exoplanet detection tool: light-curve simulation, photometric + spectroscopic
feature extraction, a small MLP trained with Adam, and a simple atmospheric
gas-retrieval fit.

Requires: numpy
    pip install numpy --break-system-packages
"""

import math
import random
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Gas database (absorption band centers in microns, width, biosignature flag)
# ──────────────────────────────────────────────────────────────────────────
GASES = [
    {"id": "H2O", "label": "H2O",  "name": "Water vapour",      "centers": [0.72, 0.94, 1.38, 1.87, 2.70, 6.27], "width": 0.055, "biosig": True},
    {"id": "CO2", "label": "CO2",  "name": "Carbon dioxide",    "centers": [1.05, 1.60, 4.26, 14.99],            "width": 0.065, "biosig": False},
    {"id": "CH4", "label": "CH4",  "name": "Methane",           "centers": [1.67, 2.31, 3.31, 7.66],             "width": 0.055, "biosig": True},
    {"id": "O3",  "label": "O3",   "name": "Ozone",             "centers": [0.60, 9.60, 14.27],                  "width": 0.060, "biosig": True},
    {"id": "O2",  "label": "O2",   "name": "Oxygen",            "centers": [0.688, 0.762, 1.27],                 "width": 0.012, "biosig": True},
    {"id": "N2O", "label": "N2O",  "name": "Nitrous oxide",     "centers": [2.87, 3.90, 7.78, 16.90],            "width": 0.050, "biosig": True},
    {"id": "DMS", "label": "DMS",  "name": "Dimethyl sulphide", "centers": [3.32, 9.50],                         "width": 0.040, "biosig": True},
    {"id": "SO2", "label": "SO2",  "name": "Sulphur dioxide",   "centers": [7.30, 8.70, 19.30],                  "width": 0.060, "biosig": False},
]

# ──────────────────────────────────────────────────────────────────────────
# Star database (example host stars / planet candidates)
# ──────────────────────────────────────────────────────────────────────────
STARS = {
    "trappist1e": {
        "name": "TRAPPIST-1 e", "period": 6.10, "depth": 0.00497, "radius": 0.92,
        "teq": 251, "esi": 0.97, "fpProb": 0.004, "planetProb": 0.996,
        "gasAmp": {"H2O": .0018, "CO2": .0008, "CH4": .0006, "O3": .0009,
                   "O2": .0004, "N2O": .0003, "DMS": .0002, "SO2": .0001},
    },
    "k218b": {
        "name": "K2-18 b", "period": 32.94, "depth": 0.0276, "radius": 2.61,
        "teq": 265, "esi": 0.63, "fpProb": 0.008, "planetProb": 0.992,
        "gasAmp": {"H2O": .0025, "CO2": .0012, "CH4": .0014, "O3": .0004,
                   "O2": .0002, "N2O": .0002, "DMS": .0008, "SO2": .0001},
    },
    "toi700d": {
        "name": "TOI-700 d", "period": 37.42, "depth": 0.00486, "radius": 1.14,
        "teq": 268, "esi": 0.89, "fpProb": 0.021, "planetProb": 0.979,
        "gasAmp": {"H2O": .0015, "CO2": .0009, "CH4": .0004, "O3": .0006,
                   "O2": .0003, "N2O": .0002, "DMS": .0001, "SO2": .0001},
    },
    "kepler90h": {
        "name": "Kepler-90 h", "period": 14.44, "depth": 0.0089, "radius": 11.32,
        "teq": 163, "esi": 0.08, "fpProb": 0.052, "planetProb": 0.948,
        "gasAmp": {"H2O": .0003, "CO2": .0008, "CH4": .0001, "O3": .0001,
                   "O2": .0001, "N2O": .0001, "DMS": .0001, "SO2": .0004},
    },
}

N = 201    # light-curve samples
NS = 280   # spectrum samples


# ──────────────────────────────────────────────────────────────────────────
# MLP with Adam optimizer (mirrors the JS Mat/MLP classes)
# ──────────────────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, n_in, h1, h2):
        s1, s2, s3 = math.sqrt(2 / n_in), math.sqrt(2 / h1), math.sqrt(2 / h2)
        self.W = [
            np.random.uniform(-s1, s1, (n_in, h1)).astype(np.float32),
            np.random.uniform(-s2, s2, (h1, h2)).astype(np.float32),
            np.random.uniform(-s3, s3, (h2, 1)).astype(np.float32),
        ]
        self.b = [np.zeros((1, h1), np.float32), np.zeros((1, h2), np.float32), np.zeros((1, 1), np.float32)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.cache = []

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def drelu(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.cache = [X]
        A = X
        for l in range(2):
            Z = A @ self.W[l] + self.b[l]
            A = self.relu(Z)
            self.cache += [Z, A]
        Zo = A @ self.W[2] + self.b[2]
        Ao = self.sigmoid(Zo)
        self.cache += [Zo, Ao]
        return Ao

    def backward(self, Y, lr):
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        m = Y.shape[0]
        Ao = self.cache[-1]
        up = (Ao - Y) / m
        for l in range(2, -1, -1):
            Ap = self.cache[0] if l == 0 else self.cache[l * 2]
            dW = Ap.T @ up
            db = up.sum(axis=0, keepdims=True)

            def adam(W, dW, mW, vW):
                mW[:] = b1 * mW + (1 - b1) * dW
                vW[:] = b2 * vW + (1 - b2) * dW * dW
                mh = mW / (1 - b1 ** self.t)
                vh = vW / (1 - b2 ** self.t)
                return W - lr * mh / (np.sqrt(vh) + eps)

            self.W[l] = adam(self.W[l], dW, self.mW[l], self.vW[l])
            self.b[l] = adam(self.b[l], db, self.mb[l], self.vb[l])
            if l > 0:
                Z = self.cache[l * 2 - 1]
                up = (up @ self.W[l].T) * self.drelu(Z)

    def train_batch(self, Xb, yb, lr):
        X = np.asarray(Xb, dtype=np.float32)
        Y = np.asarray(yb, dtype=np.float32).reshape(-1, 1)
        pred = self.forward(X)
        p = np.clip(pred, 1e-9, 1 - 1e-9)
        loss = float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())
        acc = float(((pred > 0.5) == (Y > 0.5)).mean())
        self.backward(Y, lr)
        return loss, acc

    def predict(self, x):
        X = np.asarray([x], dtype=np.float32)
        return float(self.forward(X)[0, 0])


# ──────────────────────────────────────────────────────────────────────────
# Light-curve simulation
# ──────────────────────────────────────────────────────────────────────────
def gen_lc(star):
    """Generate a realistic light curve for a known star (with its transit)."""
    f = []
    tc = int(N * 0.5)
    tw = max(8, round(N * 0.12))
    for i in range(N):
        v = (1 + 0.0015 * math.sin(2 * math.pi * i / N * 3.2)
               + 0.0008 * math.sin(2 * math.pi * i / N * 7.8 + 1.1)
               + (random.random() - 0.5) * 2 * 0.0011)
        dx = i - tc
        if abs(dx) <= tw:
            v -= star["depth"] * max(0, 1 - dx * dx / (tw * tw) * 1.25)
        f.append(v)
    return f


def gen_synth_lc(has_transit):
    """Generate a synthetic light curve for training (transit or false positive)."""
    noise_lvl = 0.0008 + random.random() * 0.002
    f = [1 + (0.001 + random.random() * 0.003) * math.sin(2 * math.pi * i / N * (1.5 + random.random() * 5))
         + (random.random() - 0.5) * 2 * noise_lvl for i in range(N)]

    if has_transit:
        d = 0.0008 + random.random() * 0.04
        c = int(N * (0.3 + random.random() * 0.4))
        w = 5 + int(random.random() * 22)
        for i in range(N):
            dx = i - c
            if abs(dx) <= w:
                f[i] -= d * max(0, 1 - dx * dx / (w * w) * 1.25)
    else:
        t = int(random.random() * 3)
        if t == 0:  # flare
            c = int(random.random() * N)
            a = 0.002 + random.random() * 0.015
            for i in range(max(0, c - 2), min(N, c + 20)):
                x = i - c
                f[i] += a * math.exp(-x * x / 20) * (1 if x >= 0 else 0.2)
        elif t == 1:  # eclipsing-binary-like V shape
            c = int(N * (0.3 + random.random() * 0.4))
            d = 0.001 + random.random() * 0.025
            w = 15 + int(random.random() * 30)
            for i in range(max(0, c - w), min(N, c + w)):
                f[i] -= d * (1 - abs(i - c) / w)
        # t == 2: pure noise, no injected signal
    return f


def noise(f):
    srt = sorted(f)
    m = srt[len(f) // 2]
    return sum(abs(v - m) for v in f) / len(f)


# ──────────────────────────────────────────────────────────────────────────
# Spectrum simulation
# ──────────────────────────────────────────────────────────────────────────
def gauss(wls, c, w, a):
    return [a * math.exp(-((v - c) ** 2) / (2 * w * w)) for v in wls]


def gen_spec_from_star(star):
    wls = [0.5 + (20 - 0.5) * i / (NS - 1) for i in range(NS)]
    depth = [star["depth"]] * NS
    for i, w in enumerate(wls):
        depth[i] += star["depth"] * 0.0007 * (w / 1.0) ** -3.5 * min(w, 1.5) / 1.5
    for g in GASES:
        a = star.get("gasAmp", {}).get(g["id"], 0)
        for c in g["centers"]:
            for i, v in enumerate(gauss(wls, c, g["width"], a)):
                depth[i] += v
    n = star["depth"] / 45
    depth = [d + (random.random() - 0.5) * 2 * n for d in depth]
    return {"wls": wls, "depth": depth}


def retrieve_atmos(spec):
    """Fit a continuum + gas-template linear model to the spectrum (gradient descent)."""
    wls, depth = spec["wls"], spec["depth"]
    n = len(wls)
    basis = []
    for g in GASES:
        col = [0.0] * n
        for c in g["centers"]:
            for i, w in enumerate(wls):
                col[i] += math.exp(-((w - c) ** 2) / (2 * g["width"] ** 2))
        basis.append(col)
    cont = [1.0] * n
    all_basis = [cont] + basis
    K = len(all_basis)
    co = [0.001] * K

    for _ in range(800):
        model_vals = [sum(co[k] * all_basis[k][i] for k in range(K)) for i in range(n)]
        res = [depth[i] - model_vals[i] for i in range(n)]
        grad = [-sum(res[i] * all_basis[k][i] for i in range(n)) / n for k in range(K)]
        co = [c - 0.0001 * grad[k] if k == 0 else max(0, c - 0.0001 * grad[k]) for k, c in enumerate(co)]

    model_vals = [sum(co[k] * all_basis[k][i] for k in range(K)) for i in range(n)]
    md = sum(depth) / n
    ss_r = sum((depth[i] - model_vals[i]) ** 2 for i in range(n))
    ss_t = sum((d - md) ** 2 for d in depth)
    r2 = 1 - ss_r / ss_t if ss_t else 0
    nz = math.sqrt(ss_r / n)

    gr = []
    for i, g in enumerate(GASES):
        amp = co[i + 1]
        vmr = amp * 1e6 * 2.5
        gr.append({
            **g, "amplitude": amp, "vmr": vmr,
            "detected": amp > md * 0.003,
            "sig": min(amp / (nz + 1e-9), 10),
        })
    return {"gr": gr, "r2": r2, "model": model_vals}


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────
def photo_f(raw):
    """20 photometric features from a light curve (transit shape/depth/SNR stats)."""
    n = len(raw)
    srt = sorted(raw)
    med = srt[n // 2] or 1
    f = [v / med for v in raw]
    mn, mx = min(f), max(f)
    rng = (mx - mn) or 1e-9
    mean = sum(f) / n
    mi = f.index(mn)

    thresh = mean - rng * 0.28
    wid = sum(1 for v in f if v < thresh) / n

    L, R = list(reversed(f[:mi])), f[mi + 1:]
    sl = min(len(L), len(R), 25)
    sym = 0.0
    for i in range(sl):
        lv = L[i] if i < len(L) else mean
        rv = R[i] if i < len(R) else mean
        sym += abs(lv - rv)
    sym = 1 - sym / (sl * rng) if sl > 0 else 0

    tri = f[max(0, mi - 8):min(n, mi + 8)]
    flat = 1 - (max(tri) - min(tri)) / rng if len(tri) > 2 else 0

    out = [f[i] for i in range(n) if abs(i - mi) > n * 0.15]
    om = sum(out) / len(out) if out else 0
    nz = math.sqrt(sum((v - om) ** 2 for v in out) / len(out)) if out else 0
    dep = mean - mn
    snr = dep / nz if nz > 0 else 0

    si = round((mi + n / 2) % n)
    sw = f[max(0, si - 7):min(n, si + 7)]
    sd = mean - min(sw) if sw else 0
    sr = sd / dep if dep > 0 else 0

    pr = f[max(0, mi - 8):max(0, mi - 3)]
    at = f[max(0, mi - 3):min(n, mi + 3)]
    pr_m = sum(pr) / len(pr) if pr else mean
    at_m = sum(at) / len(at) if at else mean
    sh = abs(pr_m - at_m) / rng

    dfs = [v - mean for v in f]
    sk = sum(v ** 3 for v in dfs) / (n * (nz ** 3 + 1e-9))
    kt = sum(v ** 4 for v in dfs) / (n * (nz ** 4 + 1e-9))

    p5 = srt[int(n * 0.05)]
    p10 = srt[int(n * 0.1)]
    p25 = srt[int(n * 0.25)]
    p75 = srt[int(n * 0.75)]

    wts = [max(0, mean - v) for v in f]
    ws = sum(wts) or 1
    ct = sum(w * i / n for i, w in enumerate(wts)) / ws

    def clip01(v):
        return max(0.0, min(1.0, v))

    return [
        dep, wid, clip01(sym), clip01(flat), min(snr / 40, 1), min(nz * 100, 1),
        min(sr / 1.5, 1), min(sh, 1), min(abs(sk) / 8, 1), min(kt / 50, 1),
        mean - p5, mean - p10, min((p75 - p25) * 10, 1), mi / n, abs(ct - 0.5) * 2,
        min(rng * 20, 1), min(sym * min(snr / 15, 1), 1), min(dep * snr / 5, 1),
        1 - min(sr, 1), min(flat * min(snr / 10, 1), 1),
    ]


def spec_f(gr):
    """12 spectroscopic features from retrieved gas amplitudes."""
    if not gr:
        return [0.0] * 12
    g = {r["id"]: r["amplitude"] for r in gr}
    e = 1e-9

    def gv(k):
        return g.get(k, 0)

    return [
        min(gv("H2O") * 200, 1), min(gv("CO2") * 200, 1), min(gv("CH4") * 200, 1),
        min(gv("O3") * 200, 1), min(gv("O2") * 200, 1), min(gv("N2O") * 200, 1),
        min(gv("DMS") * 200, 1), min(gv("SO2") * 200, 1),
        min(gv("CH4") / (gv("CO2") + e) * 2, 1),
        min(gv("O3") / (gv("H2O") + e) * 3, 1),
        min(gv("O2") / (gv("SO2") + e) / 10, 1),
        min(gv("DMS") * 500, 1),
    ]


def all_f(lc, gr):
    return photo_f(lc) + spec_f(gr)


def synth_f(has_transit):
    """32 synthetic features (20 photometric + 12 pseudo-spectroscopic) for training."""
    p = photo_f(gen_synth_lc(has_transit))
    if has_transit:
        s = [random.random() * 0.3 + 0.05 for _ in GASES] + \
            [random.random() * 0.5, random.random() * 0.4, random.random() * 0.6, random.random() * 0.3]
    else:
        s = [random.random() * 0.04 for _ in range(12)]
    return p + s[:12]


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def train_model(n_samples=700, n_epochs=40, batch_size=32, lr=1e-3, use_spectroscopy=False, verbose=True):
    """Train an MLP on synthetic light curves (half transits, half noise/flares/EBs)."""
    n_features = 32 if use_spectroscopy else 20
    h1, h2 = (48, 24) if use_spectroscopy else (32, 16)

    if verbose:
        print(f"Generating {n_samples} synthetic light curves with {n_features} features each...")

    X, y = [], []
    for i in range(n_samples):
        has_transit = (i % 2 == 0)
        if use_spectroscopy:
            feats = synth_f(has_transit)
        else:
            feats = photo_f(gen_synth_lc(has_transit))
        X.append(feats[:n_features])
        y.append(1 if has_transit else 0)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = [c[0] for c in combined], [c[1] for c in combined]

    vN = round(n_samples * 0.2)
    Xtr, ytr = X[:n_samples - vN], y[:n_samples - vN]
    Xva, yva = X[n_samples - vN:], y[n_samples - vN:]

    if verbose:
        print(f"Training set: {len(Xtr)} samples · Validation: {len(Xva)} samples")

    model = MLP(n_features, h1, h2)
    if verbose:
        print(f"Model: {n_features} -> {h1} -> {h2} -> 1, Adam lr={lr}, {n_epochs} epochs")

    for ep in range(n_epochs):
        idx = list(range(len(Xtr)))
        random.shuffle(idx)
        tL = tA = nb = 0
        for b in range(0, len(Xtr), batch_size):
            bi = idx[b:b + batch_size]
            bx = [Xtr[i] for i in bi]
            by = [ytr[i] for i in bi]
            loss, acc = model.train_batch(bx, by, lr)
            tL += loss
            tA += acc
            nb += 1
        tL /= nb
        tA /= nb

        vL = vA = 0
        for i in range(len(Xva)):
            p = model.predict(Xva[i])
            p = min(max(p, 1e-9), 1 - 1e-9)
            yt = yva[i]
            vL -= yt * math.log(p) + (1 - yt) * math.log(1 - p)
            vA += int((p > 0.5) == (yt > 0.5))
        if len(Xva):
            vL /= len(Xva)
            vA /= len(Xva)

        if verbose and (ep % 5 == 4 or ep == n_epochs - 1):
            print(f"Epoch {ep + 1}/{n_epochs} — loss {tL:.4f} — acc {tA * 100:.1f}% — val acc {vA * 100:.1f}%")

    return model


# ──────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────
def detect(model, star, lc_data, spec_data=None):
    has_spec = spec_data is not None
    gas_results = None
    if has_spec:
        ret = retrieve_atmos(spec_data)
        gas_results = ret["gr"]

    n_features = 32 if has_spec else 20
    feats = all_f(lc_data, gas_results)[:n_features]
    raw = model.predict(feats)
    prob = raw * 0.6 + star["planetProb"] * 0.4 if star.get("planetProb") is not None else raw

    return {
        "probability": prob,
        "verdict": "Transit detected — planet candidate" if prob > 0.5 else "No transit detected",
        "false_positive_prob": star.get("fpProb", 1 - prob),
        "mode": "Photometric + spectroscopic" if has_spec else "Photometric only",
        "gas_results": gas_results,
    }


# ──────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    star = STARS["trappist1e"]

    print(f"=== Analyzing {star['name']} ===\n")

    # Photometric-only pipeline
    model = train_model(n_samples=700, n_epochs=30, batch_size=32, lr=1e-3, use_spectroscopy=False)
    lc = gen_lc(star)
    result = detect(model, star, lc)

    print("\n--- Detection result (photometric) ---")
    print(f"Planet probability: {result['probability'] * 100:.1f}%")
    print(f"Verdict: {result['verdict']}")
    print(f"False positive probability: {result['false_positive_prob'] * 100:.1f}%")

    # Photometric + spectroscopic pipeline
    print(f"\n=== Adding spectroscopy ===\n")
    spec = gen_spec_from_star(star)
    model_hs = train_model(n_samples=700, n_epochs=30, batch_size=32, lr=1e-3, use_spectroscopy=True)
    result_hs = detect(model_hs, star, lc, spec)

    print("\n--- Detection result (photometric + spectroscopic) ---")
    print(f"Planet probability: {result_hs['probability'] * 100:.1f}%")
    print(f"Verdict: {result_hs['verdict']}")
    print("\nGas retrieval:")
    for g in result_hs["gas_results"]:
        flag = " (biosignature)" if g["biosig"] and g["detected"] else ""
        status = f"{g['vmr']:.2f} ppm, {g['sig']:.1f}sigma{flag}" if g["detected"] else "below detection limit"
        print(f"  {g['label']:>4} ({g['name']:<18}): {status}")
