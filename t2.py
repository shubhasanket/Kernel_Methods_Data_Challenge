import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## Kindly refer to the Readme for instructions

## We hardcode the best hyperparameters found by running t1.py
## And simply ensemble them and run the method on the test set

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XTR_PATH = os.path.join(BASE_DIR, "Xtr.csv")
XTE_PATH = os.path.join(BASE_DIR, "Xte.csv")
YTR_PATH = os.path.join(BASE_DIR, "Ytr.csv")

SUBMISSION_PATH = os.path.join(BASE_DIR, "Yte.csv")


def load_data():
    Xtr = np.array(pd.read_csv(XTR_PATH, header=None, usecols=range(3072)), dtype=np.float32)
    Xte = np.array(pd.read_csv(XTE_PATH, header=None, usecols=range(3072)), dtype=np.float32)
    Ytr = np.array(pd.read_csv(YTR_PATH, usecols=[1])).squeeze().astype(np.int64)
    return Xtr, Xte, Ytr

def flat_to_img(X):
    return X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)

def normalize_rows(X, eps=1e-8):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)

def stratified_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


# preprocessing steps
def per_image_standardize(imgs):
    mean = imgs.mean(axis=(1, 2, 3), keepdims=True)
    std = imgs.std(axis=(1, 2, 3), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (imgs - mean) / std

def rgb_to_gray(imgs):
    return (
        0.299 * imgs[..., 0] +
        0.587 * imgs[..., 1] +
        0.114 * imgs[..., 2]
    ).astype(np.float32)



def sq_dists(A, B):
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)

def median_gamma(X, sample_size=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
    Z = X[idx]
    D = sq_dists(Z, Z)
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return 1.0
    return 1.0 / (np.median(vals) + 1e-12)

# HOG
def compute_gradients(gray):
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    return gx, gy


# HOG helper functions
def _hog_hist_tensor(gray, cell=4, bins=9, signed=False):
    gx, gy = compute_gradients(gray)
    mag = np.sqrt(gx * gx + gy * gy)

    if signed:
        ori = np.mod(np.arctan2(gy, gx), 2.0 * np.pi)
        max_angle = 2.0 * np.pi
    else:
        ori = np.mod(np.arctan2(gy, gx), np.pi)
        max_angle = np.pi

    n, h, w = gray.shape
    gh, gw = h // cell, w // cell
    hist = np.zeros((n, gh, gw, bins), dtype=np.float32)
    bin_width = max_angle / bins

    for i in range(gh):
        for j in range(gw):
            m = mag[:, i*cell:(i+1)*cell, j*cell:(j+1)*cell].reshape(n, -1)
            o = ori[:, i*cell:(i+1)*cell, j*cell:(j+1)*cell].reshape(n, -1)

            b = np.floor(o / bin_width).astype(np.int32)
            b = np.clip(b, 0, bins - 1)

            for k in range(bins):
                hist[:, i, j, k] = np.sum(m * (b == k), axis=1)

    return hist

def _safe_grid_pool(hist, grid):
    n, gh, gw, bins = hist.shape
    feats = []

    for i in range(grid):
        for j in range(grid):
            h0 = i * gh // grid
            h1 = (i + 1) * gh // grid
            w0 = j * gw // grid
            w1 = (j + 1) * gw // grid

            if h1 <= h0 or w1 <= w0:
                continue

            block = hist[:, h0:h1, w0:w1, :]
            feats.append(block.mean(axis=(1, 2)))

    if len(feats) == 0:
        feats.append(hist.mean(axis=(1, 2)))

    return np.concatenate(feats, axis=1)

def _spatial_pyramid_pool(hist):
    n, gh, gw, bins = hist.shape
    feats = [hist.mean(axis=(1, 2))]

    if gh >= 2 and gw >= 2:
        feats.append(_safe_grid_pool(hist, 2))
    if gh >= 4 and gw >= 4:
        feats.append(_safe_grid_pool(hist, 4))

    F = np.concatenate(feats, axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def extract_hog_spm(imgs, cell=4, bins=9, signed=False):
    gray = rgb_to_gray(imgs)
    hist = _hog_hist_tensor(gray, cell=cell, bins=bins, signed=signed)
    feat = _spatial_pyramid_pool(hist)
    return normalize_rows(feat).astype(np.float32)

# color features
def extract_color_stats(imgs, cell=8):
    n, h, w, c = imgs.shape
    gh, gw = h // cell, w // cell
    feats = []
    for i in range(gh):
        for j in range(gw):
            patch = imgs[:, i*cell:(i+1)*cell, j*cell:(j+1)*cell, :]
            feats.append(patch.mean(axis=(1, 2)))
            feats.append(patch.std(axis=(1, 2)))
    F = np.concatenate(feats, axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_rows(F).astype(np.float32)

# 3 layer CKN
class SampledCKNLayer:
    def __init__(self, patch_size=5, n_filters=48, n_patch_samples=16000,
                 stride=1, subsample=2, seed=0):
        self.patch_size = patch_size
        self.n_filters = n_filters
        self.n_patch_samples = n_patch_samples
        self.stride = stride
        self.subsample = subsample
        self.seed = seed

    def _all_patch_positions(self, h, w):
        ps = self.patch_size
        positions = []
        for i in range(0, h - ps + 1, self.stride):
            for j in range(0, w - ps + 1, self.stride):
                positions.append((i, j))
        return positions

    def _extract_sampled_patches_for_fit(self, maps):
        rng = np.random.default_rng(self.seed)
        n, h, w, c = maps.shape
        positions = self._all_patch_positions(h, w)

        total_possible = n * len(positions)
        sample_n = min(self.n_patch_samples, total_possible)

        img_ids = rng.integers(0, n, size=sample_n)
        pos_ids = rng.integers(0, len(positions), size=sample_n)

        P = np.empty((sample_n, self.patch_size * self.patch_size * c), dtype=np.float32)
        for t in range(sample_n):
            ii = img_ids[t]
            i, j = positions[pos_ids[t]]
            P[t] = maps[ii, i:i+self.patch_size, j:j+self.patch_size, :].reshape(-1)

        P = P - P.mean(axis=1, keepdims=True)
        P = normalize_rows(P)
        return np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_dense_patches(self, maps):
        n, h, w, c = maps.shape
        out_h = 1 + (h - self.patch_size) // self.stride
        out_w = 1 + (w - self.patch_size) // self.stride

        patches = np.empty((n, out_h, out_w, self.patch_size * self.patch_size * c), dtype=np.float32)
        oi = 0
        for i in range(0, h - self.patch_size + 1, self.stride):
            oj = 0
            for j in range(0, w - self.patch_size + 1, self.stride):
                patches[:, oi, oj, :] = maps[:, i:i+self.patch_size, j:j+self.patch_size, :].reshape(n, -1)
                oj += 1
            oi += 1
        return patches

    def fit(self, maps):
        rng = np.random.default_rng(self.seed + 1)
        P = self._extract_sampled_patches_for_fit(maps)

        idx = rng.choice(P.shape[0], size=min(self.n_filters, P.shape[0]), replace=False)
        self.anchors_ = P[idx].astype(np.float32)

        ref = P[:min(4000, len(P))]
        D = sq_dists(ref, self.anchors_)
        vals = D[D > 1e-12]
        self.gamma_ = 1.0 / (np.median(vals) + 1e-12) if len(vals) > 0 else 1.0

        W = np.exp(-self.gamma_ * sq_dists(self.anchors_, self.anchors_)).astype(np.float32)
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        W += 1e-5 * np.eye(W.shape[0], dtype=np.float32)

        U, S, _ = np.linalg.svd(W, full_matrices=False)
        self.W_inv_sqrt_ = (U @ np.diag(1.0 / np.sqrt(S + 1e-6))).astype(np.float32)
        return self

    def transform(self, maps):
        patches = self._extract_dense_patches(maps)
        n, h, w, d = patches.shape

        P = patches.reshape(-1, d).astype(np.float32)
        P = P - P.mean(axis=1, keepdims=True)
        P = normalize_rows(P)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        K = np.exp(-self.gamma_ * sq_dists(P, self.anchors_)).astype(np.float32)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

        Z = K @ self.W_inv_sqrt_
        Z = Z.reshape(n, h, w, -1)

        s = self.subsample
        out_h = h // s
        out_w = w // s
        if out_h > 0 and out_w > 0:
            Z = Z[:, :out_h*s, :out_w*s, :]
            Z = Z.reshape(n, out_h, s, out_w, s, Z.shape[-1]).mean(axis=(2, 4))
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z.astype(np.float32)


# CKN
class Strong3LayerCKN:
    def __init__(self, l1_filters=80, l2_filters=112, l3_filters=144, seed=0):
        self.l1 = SampledCKNLayer(
            patch_size=5, n_filters=l1_filters, n_patch_samples=18000,
            stride=1, subsample=2, seed=seed + 1
        )
        self.l2 = SampledCKNLayer(
            patch_size=2, n_filters=l2_filters, n_patch_samples=14000,
            stride=1, subsample=2, seed=seed + 2
        )
        self.l3 = SampledCKNLayer(
            patch_size=2, n_filters=l3_filters, n_patch_samples=10000,
            stride=1, subsample=1, seed=seed + 3
        )

    def fit(self, imgs):
        gray = rgb_to_gray(imgs)[..., None]
        self.l1.fit(gray)
        z1 = self.l1.transform(gray)
        self.l2.fit(z1)
        z2 = self.l2.transform(z1)
        self.l3.fit(z2)
        return self

    def transform(self, imgs):
        gray = rgb_to_gray(imgs)[..., None]
        z1 = self.l1.transform(gray)
        z2 = self.l2.transform(z1)
        z3 = self.l3.transform(z2)
        F = z3.mean(axis=(1, 2))
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_rows(F).astype(np.float32)

# Nystrom / RFF
class Nystrom:
    def __init__(self, m=160, seed=0):
        self.m = m
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        m = min(self.m, len(X))
        idx = rng.choice(len(X), m, replace=False)
        self.landmarks = X[idx].astype(np.float32)
        self.gamma = median_gamma(self.landmarks, seed=self.seed)

        K = np.exp(-self.gamma * sq_dists(self.landmarks, self.landmarks)).astype(np.float32)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        K += 1e-5 * np.eye(K.shape[0], dtype=np.float32)

        U, S, _ = np.linalg.svd(K, full_matrices=False)
        self.W = (U @ np.diag(1.0 / np.sqrt(S + 1e-6))).astype(np.float32)
        return self

    def transform(self, X):
        K = np.exp(-self.gamma * sq_dists(X, self.landmarks)).astype(np.float32)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_rows(K @ self.W).astype(np.float32)

class RFF:
    def __init__(self, D=160, seed=0):
        self.D = D
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        gamma = median_gamma(X, seed=self.seed)
        self.W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(X.shape[1], self.D)).astype(np.float32)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=(self.D,)).astype(np.float32)
        return self

    def transform(self, X):
        Z = np.sqrt(2.0 / self.D) * np.cos(X @ self.W + self.b[None, :])
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_rows(Z).astype(np.float32)

# MKL + rridge
def one_hot(y):
    return np.eye(len(np.unique(y)))[y]

def compute_weights(features, y, temperature=1.0):
    Y = one_hot(y)
    Ky = Y @ Y.T

    keys = list(features.keys())
    vals = []
    for k in keys:
        K = features[k] @ features[k].T
        score = np.sum(K * Ky)
        vals.append(max(score, 0.0))

    vals = np.array(vals, dtype=np.float64)
    vals = np.power(vals + 1e-12, temperature)
    vals /= (vals.sum() + 1e-12)
    return dict(zip(keys, vals))

def combine(features, weights):
    Z = np.concatenate([np.sqrt(weights[k]) * features[k] for k in weights], axis=1)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_rows(Z).astype(np.float32)

class Ridge:
    def __init__(self, lam=0.01):
        self.lam = lam

    def fit(self, Z, y):
        Y = one_hot(y).astype(np.float32)
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1], dtype=np.float32)
        B = Z.T @ Y
        self.W = np.linalg.solve(A, B)

    def decision_function(self, Z):
        return Z @ self.W

    def predict(self, Z):
        return np.argmax(self.decision_function(Z), axis=1)

# final feature building class
class FeatureBuilder:
    def __init__(
        self,
        l1_filters=80,
        l2_filters=112,
        l3_filters=144,
        nystrom_m=180,
        rff_D=256,
        seed=0
    ):
        self.l1_filters = l1_filters
        self.l2_filters = l2_filters
        self.l3_filters = l3_filters
        self.nystrom_m = nystrom_m
        self.rff_D = rff_D
        self.seed = seed

    def fit(self, X):
        imgs = flat_to_img(X / 255.0)
        imgs = per_image_standardize(imgs)

        hog4_u9 = extract_hog_spm(imgs, cell=4, bins=9, signed=False)
        hog4_u18 = extract_hog_spm(imgs, cell=4, bins=18, signed=False)
        hog4_s18 = extract_hog_spm(imgs, cell=4, bins=18, signed=True)
        hog8_u9 = extract_hog_spm(imgs, cell=8, bins=9, signed=False)
        hog8_s18 = extract_hog_spm(imgs, cell=8, bins=18, signed=True)
        hog16_u9 = extract_hog_spm(imgs, cell=16, bins=9, signed=False)
        color = extract_color_stats(imgs, cell=8)

        self.ckn = Strong3LayerCKN(
            l1_filters=self.l1_filters,
            l2_filters=self.l2_filters,
            l3_filters=self.l3_filters,
            seed=self.seed + 10
        )
        self.ckn.fit(imgs)
        cknf = self.ckn.transform(imgs)

        self.hog4_u9_map = Nystrom(m=self.nystrom_m, seed=self.seed + 1).fit(hog4_u9)
        self.hog4_u18_map = Nystrom(m=self.nystrom_m, seed=self.seed + 2).fit(hog4_u18)
        self.hog4_s18_map = Nystrom(m=self.nystrom_m, seed=self.seed + 3).fit(hog4_s18)
        self.hog8_u9_map = Nystrom(m=max(140, self.nystrom_m - 20), seed=self.seed + 4).fit(hog8_u9)
        self.hog8_s18_map = Nystrom(m=max(140, self.nystrom_m - 20), seed=self.seed + 5).fit(hog8_s18)
        self.hog16_u9_map = Nystrom(m=max(100, self.nystrom_m - 60), seed=self.seed + 6).fit(hog16_u9)

        self.ckn_map = Nystrom(m=max(140, self.nystrom_m - 20), seed=self.seed + 7).fit(cknf)
        self.color_map = RFF(D=self.rff_D, seed=self.seed + 8).fit(color)
        return self

    def transform(self, X):
        imgs = flat_to_img(X / 255.0)
        imgs = per_image_standardize(imgs)

        hog4_u9 = extract_hog_spm(imgs, cell=4, bins=9, signed=False)
        hog4_u18 = extract_hog_spm(imgs, cell=4, bins=18, signed=False)
        hog4_s18 = extract_hog_spm(imgs, cell=4, bins=18, signed=True)
        hog8_u9 = extract_hog_spm(imgs, cell=8, bins=9, signed=False)
        hog8_s18 = extract_hog_spm(imgs, cell=8, bins=18, signed=True)
        hog16_u9 = extract_hog_spm(imgs, cell=16, bins=9, signed=False)
        color = extract_color_stats(imgs, cell=8)
        cknf = self.ckn.transform(imgs)

        return {
            "hog4_u9": self.hog4_u9_map.transform(hog4_u9),
            "hog4_u18": self.hog4_u18_map.transform(hog4_u18),
            "hog4_s18": self.hog4_s18_map.transform(hog4_s18),
            "hog8_u9": self.hog8_u9_map.transform(hog8_u9),
            "hog8_s18": self.hog8_s18_map.transform(hog8_s18),
            "hog16_u9": self.hog16_u9_map.transform(hog16_u9),
            "ckn": self.ckn_map.transform(cknf),
            "color": self.color_map.transform(color),
        }


def fit_predict_single_config(Xtr, ytr, Xte, cfg):
    builder = FeatureBuilder(
        l1_filters=cfg["l1_filters"],
        l2_filters=cfg["l2_filters"],
        l3_filters=cfg["l3_filters"],
        nystrom_m=cfg["nystrom_m"],
        rff_D=cfg["rff_D"],
        seed=cfg["seed"]
    )

    builder.fit(Xtr)

    Ftr = builder.transform(Xtr)
    weights = compute_weights(Ftr, ytr, temperature=cfg["mkl_temp"])
    Ztr = combine(Ftr, weights)

    clf = Ridge(lam=cfg["lam"])
    clf.fit(Ztr, ytr)

    del Ftr, Ztr
    gc.collect()

    Fte = builder.transform(Xte)
    Zte = combine(Fte, weights)
    scores = clf.decision_function(Zte)

    del builder, Fte, Zte, clf
    gc.collect()

    return scores

def ensemble_predict_weighted(Xtr, ytr, Xte, cfgs, weights):
    total_scores = None

    for i, (cfg, w) in enumerate(zip(cfgs, weights)):
        print(f"\nEnsemble model {i+1}/{len(cfgs)} | weight={w:.3f}")
        print(cfg)

        scores = fit_predict_single_config(Xtr, ytr, Xte, cfg)

        if total_scores is None:
            total_scores = w * scores
        else:
            total_scores += w * scores

        gc.collect()

    return np.argmax(total_scores, axis=1)

def ensemble_predict(Xtr, ytr, Xte, cfgs):
    total_scores = None
    for i, cfg in enumerate(cfgs):
        print(f"Final ensemble model {i+1}/{len(cfgs)}: {cfg}")
        scores = fit_predict_single_config(Xtr, ytr, Xte, cfg)
        if total_scores is None:
            total_scores = scores
        else:
            total_scores += scores
    return np.argmax(total_scores, axis=1)


def main():

    Xtr, Xte, Ytr = load_data()

    # These confings were obtained by running the file t1.py
    # We simply perform the ensembling over the best configs
    # and run the model on the test set

    best_cfgs = [
        {
            "l1_filters": 96, "l2_filters": 128, "l3_filters": 128,
            "nystrom_m": 260, "rff_D": 256,
            "lam": 0.04, "mkl_temp": 1.5, "seed": 1
        },
        {
            "l1_filters": 112, "l2_filters": 144, "l3_filters": 144,
            "nystrom_m": 260, "rff_D": 256,
            "lam": 0.04, "mkl_temp": 1.5, "seed": 2
        },
        {
            "l1_filters": 80, "l2_filters": 128, "l3_filters": 128,
            "nystrom_m": 260, "rff_D": 256,
            "lam": 0.03, "mkl_temp": 1.5, "seed": 3
        }
    ]

    weights = [1/3, 1/3, 1/3]

    # Validation check
    print("\nRunning validation check...")

    X_train, y_train, X_val, y_val = stratified_split(Xtr, Ytr, val_ratio=0.2, seed=42)

    total_scores = None

    for i, (cfg, w) in enumerate(zip(best_cfgs, weights)):
        print(f"\nValidation model {i+1}/{len(best_cfgs)}")
        scores = fit_predict_single_config(X_train, y_train, X_val, cfg)

        if total_scores is None:
            total_scores = w * scores
        else:
            total_scores += w * scores

    val_pred = np.argmax(total_scores, axis=1)
    val_acc = np.mean(val_pred == y_val)

    print("\nValidation accuracy:", val_acc)

    cm = confusion_matrix(y_val, val_pred)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    print("\nConfusion matrix (counts):")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, cmap='Blues')
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(fraction=0.046, pad=0.04)

    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    plt.xticks(range(n_classes))
    plt.yticks(range(n_classes))
    plt.tight_layout()
    plt.show()
    # Final training on full data
    print("\nRunning final weighted ensemble on full data...")

    Yte_pred = ensemble_predict_weighted(Xtr, Ytr, Xte, best_cfgs, weights)

    # Save submission

    submission = pd.DataFrame({"Prediction": Yte_pred.astype(int)})
    submission.index += 1
    submission.to_csv(SUBMISSION_PATH, index_label="Id")

    print("\nSaved:", SUBMISSION_PATH)

if __name__ == "__main__":
    main()