# BEN 2026-11-01
# Change aim: Synthetic prostate-biopsy navigation demo: point registration (FRE/TRE), pivot calibration, needle targeting error.
# File: navigation_demo.py
"""SIMULATED data only (no patient images). Units: millimetres.
Steps: (1) rigid point registration by SVD, with FRE and TRE measured by Monte Carlo and compared with the
Fitzpatrick-West-Maurer (1998) TRE prediction; (2) pivot calibration of a tracked needle tip;
(3) combined needle targeting error. Research/portfolio code, not a medical device."""
import argparse
import os
import numpy as np
import cv2

FLE_SIGMA = 0.5        # fiducial localisation noise per axis (mm), an assumption
POSE_SIGMA = 0.3       # tool position noise per axis (mm), an assumption
ROT_SIGMA_DEG = 0.2    # tool orientation noise (deg), an assumption
FIDUCIALS = np.array([[-55, -40, -25], [50, -45, 20], [58, 35, -22], [-48, 42, 28], [5, -5, 38], [0, 50, -30]], float)
TARGET = np.array([8.0, -5.0, 12.0])
TIP_TRUE = np.array([0.0, 0.0, -150.0])   # needle tip in tool frame
PIVOT_TRUE = np.array([100.0, -30.0, 320.0])


def euler_to_matrix(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


R_TRUE = euler_to_matrix(15, -10, 25)
T_TRUE = np.array([120.0, -40.0, 300.0])


def register_rigid(a, b):
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def fre_rms(R, t, a, b):
    d = (a @ R.T + t) - b
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def predicted_tre(a, target, fle_sigma):
    n = len(a)
    ca = a.mean(axis=0)
    q = a - ca
    w, V = np.linalg.eigh(q.T @ q)
    r = target - ca
    s = 0.0
    for k in range(3):
        v = V[:, k]
        d2 = r @ r - (r @ v) ** 2
        f2 = ((q * q).sum(axis=1) - (q @ v) ** 2).mean()
        s += d2 / f2
    fle2 = 3 * fle_sigma ** 2
    return float(np.sqrt(fle2 / n * (1 + s / 3)))


def pivot_calibration(rots, pos):
    A = np.vstack([np.hstack([R, -np.eye(3)]) for R in rots])
    b = -np.concatenate(pos)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], x[3:]


def small_rotation(rng, sigma_deg):
    v = np.radians(sigma_deg) * rng.standard_normal(3)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def calibrate_once(rng, n_poses=60):
    rots, pos = [], []
    for _ in range(n_poses):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rng.uniform(5, 40))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        p = PIVOT_TRUE - R @ TIP_TRUE
        rots.append(small_rotation(rng, ROT_SIGMA_DEG) @ R)
        pos.append(p + POSE_SIGMA * rng.standard_normal(3))
    tip, _ = pivot_calibration(rots, pos)
    return tip


def monte_carlo(seed, trials, out_dir):
    rng = np.random.default_rng(seed)
    n = len(FIDUCIALS)
    fre2, tre2, tip_err, total, trial_rows = [], [], [], [], []
    target_trk = R_TRUE @ TARGET + T_TRUE
    needle_R = euler_to_matrix(0, 30, 0)
    for i in range(trials):
        b = FIDUCIALS @ R_TRUE.T + T_TRUE + FLE_SIGMA * rng.standard_normal((n, 3))
        R, t = register_rigid(FIDUCIALS, b)
        fre2.append(fre_rms(R, t, FIDUCIALS, b) ** 2)
        e_reg = (R @ TARGET + t) - target_trk
        tre2.append(float(e_reg @ e_reg))
        tip_est = calibrate_once(rng, 30)
        e_tip = needle_R @ (TIP_TRUE - tip_est)
        tip_err.append(float(np.linalg.norm(TIP_TRUE - tip_est)))
        e_total = e_reg + e_tip
        total.append(float(np.linalg.norm(e_total)))
        trial_rows.append((i, float(np.sqrt(tre2[-1])), tip_err[-1], total[-1], *e_reg))
    total = np.array(total)
    lines = [
        f"fiducials: {n}, FLE sigma {FLE_SIGMA} mm per axis, trials {trials}",
        f"RMS FRE measured:  {np.sqrt(np.mean(fre2)):.4f} mm",
        f"RMS FRE predicted: {np.sqrt((1 - 2 / n) * 3 * FLE_SIGMA ** 2):.4f} mm",
        f"RMS TRE measured:  {np.sqrt(np.mean(tre2)):.4f} mm",
        f"RMS TRE predicted: {predicted_tre(FIDUCIALS, TARGET, FLE_SIGMA):.4f} mm",
        f"mean tip calibration error: {np.mean(tip_err):.4f} mm",
        f"targeting error mean: {total.mean():.4f} mm, median {np.median(total):.4f} mm, 95th percentile {np.percentile(total, 95):.4f} mm",
        f"within 2 mm: {100 * np.mean(total <= 2):.1f} %, within 3 mm: {100 * np.mean(total <= 3):.1f} %, within 5 mm: {100 * np.mean(total <= 5):.1f} %",
    ]
    os.makedirs(out_dir, exist_ok=True)
    text = "\n".join(lines)
    print(text)
    open(f"{out_dir}/summary_py.txt", "w").write(text + "\n")
    with open(f"{out_dir}/trials_py.csv", "w") as f:
        f.write("trial,tre_mm,tip_cal_error_mm,targeting_error_mm,ex,ey,ez\n")
        for r in trial_rows:
            f.write(",".join(f"{v:.5f}" if j else str(int(v)) for j, v in enumerate(r)) + "\n")
    return trial_rows


def check_file(path):
    img, trk, tgt = [], [], None
    for line in open(path):
        p = line.strip().split(",")
        if p[0] == "img":
            img.append([float(v) for v in p[1:]])
        elif p[0] == "trk":
            trk.append([float(v) for v in p[1:]])
        elif p[0] == "target":
            tgt = np.array([float(v) for v in p[1:]])
    a, b = np.array(img), np.array(trk)
    R, t = register_rigid(a, b)
    print("R:", " ".join(f"{v:.5f}" for v in R.ravel()))
    print("t:", " ".join(f"{v:.5f}" for v in t))
    print(f"FRE: {fre_rms(R, t, a, b):.5f}")
    print(f"target in tracker space: {' '.join(f'{v:.5f}' for v in (R @ tgt + t))}")
    print(f"predicted RMS TRE: {predicted_tre(a, tgt, FLE_SIGMA):.5f}")


def write_check(path):
    rng = np.random.default_rng(123)
    b = FIDUCIALS @ R_TRUE.T + T_TRUE + FLE_SIGMA * rng.standard_normal(FIDUCIALS.shape)
    with open(path, "w") as f:
        for p in FIDUCIALS:
            f.write("img," + ",".join(f"{v:.6f}" for v in p) + "\n")
        for p in b:
            f.write("trk," + ",".join(f"{v:.6f}" for v in p) + "\n")
        f.write("target," + ",".join(f"{v:.6f}" for v in TARGET) + "\n")


def draw_slice(path, trial):
    size, ppm = 256, 4.0
    rng = np.random.default_rng(7)
    a = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 1.2)
    b = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 1.2)
    img = np.sqrt(a * a + b * b)
    img /= img.mean()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mx, my = (xx - size / 2) / ppm, (yy - size / 2) / ppm
    gland = ((mx + 3) / 24) ** 2 + ((my - 2) / 18) ** 2
    img = np.where(gland < 1, 0.55 * img, img)
    img = np.where((gland > 0.9) & (gland < 1.05), 1.4 * img, img)
    img = np.where(mx ** 2 + my ** 2 < 4.0 ** 2, 0.25 * img, img)
    g = np.clip(img * 110, 0, 255).astype(np.uint8)
    col = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (512, 512), interpolation=cv2.INTER_CUBIC)
    ppm, half = 8.0, 256
    c = (half, half)
    est = (int(round(half + trial[4] * ppm)), int(round(half + trial[5] * ppm)))
    cv2.line(col, (c[0] + 80, 0), est, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.circle(col, c, 32, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(col, est, (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.line(col, (20, 492), (20 + int(10 * ppm), 492), (255, 255, 255), 2)
    cv2.putText(col, "10 mm", (20, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(col, "SIMULATED slice", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(col, "green: true target   red: registered estimate", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--check", default="")
    ap.add_argument("--write-check", default="")
    a = ap.parse_args()
    if a.write_check:
        write_check(a.write_check)
        return
    if a.check:
        check_file(a.check)
        return
    rows = monte_carlo(a.seed, a.trials, a.out)
    med = sorted(rows, key=lambda r: r[1])[len(rows) // 2]
    draw_slice(f"{a.out}/slice_py.png", med)


if __name__ == "__main__":
    main()
