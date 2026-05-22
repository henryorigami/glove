# MANUS to Hand V8 Retargeting

This module retargets MANUS raw skeleton CSVs to the 21 revolute joints in the
Hand V8 URDF.

Current approach:

1. Read MANUS raw skeleton frames from `manus_raw_skeleton.csv`.
2. Use one open/neutral MANUS frame as calibration.
3. Build comparable MANUS and robot keypoints for thumb/index/middle/ring/pinky.
4. Preserve the robot hand's own segment lengths.
5. Solve bounded IK per frame with temporal smoothing.

This is intentionally close to the DexPilot / DEX-Retargeting family of methods:
track fingertip and segment-vector geometry rather than directly copying human
joint angles, because the MANUS hand and robot hand do not have identical axes,
link lengths, or thumb layout.

## Run

```powershell
cd C:\Users\henry\Desktop\hand_capture
python -m retargeting.retarget_hand_v8 `
  --manus-csv C:\Users\henry\Desktop\hand_capture\recordings\session_YYYY\manus_raw_skeleton.csv `
  --out C:\Users\henry\Desktop\hand_capture\recordings\session_YYYY\retarget_hand_v8.csv `
  --max-frames 1000
```

Output CSV columns:

- `t_wall_ns`
- `manus_frame_seq`
- 21 robot joint columns in URDF order:
  `i0 i1 i2 i3 m0 m1 m2 m3 r0 r1 r2 r3 p0 p1 p2 p3 t0 t1 t2 t3 t4`
- IK diagnostics:
  `ik_cost`, `ik_success`, `ik_nfev`, `mean_tip_error_m`

The sidecar `*.summary.json` contains aggregate error and calibration stats.

## Live MuJoCo Visualization

Run the native live viewer. This starts MANUS Core Integrated in WSL itself,
streams skeleton frames directly over stdout, retargets them, and writes the 21
Hand V8 joint positions straight into MuJoCo.

```powershell
cd C:\Users\henry\Desktop\hand_capture
python -m retargeting.live_manus_mujoco_hand_v8
```

Useful options:

```powershell
python -m retargeting.live_manus_mujoco_hand_v8 `
  --solve-every 1 `
  --max-nfev 18 `
  --display-alpha 0.65 `
  --wrist-mode local
```

The live script still records MANUS CSVs under:

```text
C:\Users\henry\Desktop\hand_capture\recordings\live_mujoco_...
```

Stop the dashboard session before running this standalone viewer; otherwise two
MANUS Core Integrated instances can fight over the same dongle. The MuJoCo
viewer uses `--wrist-mode local` by default so wrist IMU rotation is not applied
twice to stationary fingers.

## Notes

- Use an open/neutral hand for the first frame, or pass `--calibration-frame N`.
- The thumb is the least trustworthy part of this first pass because the robot
  thumb has 5 DOF while the current MANUS keypoint selection has fewer directly
  comparable points.
- The live viewer defaults to a lower IK iteration budget than offline CSV
  retargeting so MuJoCo stays close to live.
