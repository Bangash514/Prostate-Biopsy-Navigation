# Prostate Biopsy Navigation: Registration, Calibration and Targeting Error (C++ and Python)

Author: BEN, Software Engineer · 2026-11-01

A simulated demo of the maths behind image-guided prostate biopsy: register image space to tracker space,
calibrate a tracked needle tip, then measure how far the needle lands from the target. Same algorithms in
C++ (Eigen + OpenCV) and Python (NumPy).

> **Simulated data only.** No patient images, no real tracker. It is research / portfolio code,
> not a medical device, and makes no clinical claims. Noise levels are assumptions, listed below.

## What it does
1. **Point-based rigid registration** (SVD method of Arun / Horn) of 6 fiducials, with fiducial registration error (FRE) and target registration error (TRE) measured over 5000 noisy trials.
2. **TRE prediction** with the formula of Fitzpatrick, West and Maurer (1998), compared with the measured TRE.
3. **Pivot calibration** of a tracked needle tip (least squares over 30 tool poses per trial).
4. **Needle targeting error**: registration error plus tip-calibration error, reported as mean, median, 95th percentile and the share of trials within 2, 3 and 5 mm.
5. A **simulated ultrasound-like slice** with the target, needle path and one registered estimate (`results/slice_cpp.png`).

## Assumptions (change them in the source)
Fiducial localisation noise 0.5 mm per axis; tool position noise 0.3 mm per axis; tool orientation noise 0.2 deg; 6 fiducials around the gland; needle tip 150 mm from the tool origin. These are illustrative values, not measurements of any device.

## Layout
- `cpp/navigation_demo.cpp` – C++ version (CMake)
- `python/navigation_demo.py` – Python version
- `tools/plot_results.py` – histogram of targeting error
- `data/check_fiducials.csv` – shared fixed input for the exact C++ vs Python check
- `results/` – summaries, per-trial CSV files, pictures

## Build and run
```bash
sudo apt install libeigen3-dev libopencv-dev cmake g++
cmake -S . -B build && cmake --build build
./build/navigation_demo --trials 5000 --out results

pip install numpy opencv-python matplotlib
python3 python/navigation_demo.py --trials 5000 --out results
python3 tools/plot_results.py

# exact comparison on a fixed input
./build/navigation_demo --check data/check_fiducials.csv
python3 python/navigation_demo.py --check data/check_fiducials.csv
```

## Measured results (5000 simulated trials)
| Quantity | C++ | Python |
|---|---|---|
| RMS FRE measured / predicted (mm) | 0.7055 / 0.7071 | 0.7067 / 0.7071 |
| RMS TRE measured / predicted (mm) | 0.3669 / 0.3654 | 0.3640 / 0.3654 |
| Mean tip calibration error (mm) | 0.4635 | 0.4619 |
| Targeting error mean / median / 95th pct (mm) | 0.578 / 0.554 / 1.023 | 0.573 / 0.548 / 1.011 |

- On the fixed check file, C++ and Python print **identical** transforms, FRE and TRE prediction (5 decimals).
- Measured and predicted FRE and TRE agree within Monte Carlo noise, which supports the implementation.
- The two languages use different random generators, so their Monte Carlo numbers differ slightly.

## Honest limits
- Simulation only. Real systems add tracker distortion, organ motion, imaging error and needle bending, which are not modelled.
- Isotropic Gaussian noise is assumed; real fiducial localisation error is not isotropic.
- The targeting error combines only registration and tip calibration; it is not a full clinical error budget.
- The 2, 3 and 5 mm columns are demo thresholds, not clinical acceptance limits.

## References
- Fitzpatrick JM, West JB, Maurer CR. Predicting error in rigid-body point-based registration. IEEE Trans Med Imaging 1998.
- Arun KS, Huang TS, Blostein SD. Least-squares fitting of two 3-D point sets. IEEE Trans Pattern Anal Mach Intell 1987.
(Check citation details before reusing.)

## License
MIT, see `LICENSE`.
