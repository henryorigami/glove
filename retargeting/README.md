# MANUS to Hand V9 Retargeting

This module retargets MANUS raw skeleton CSVs to the 21 revolute joints in the
Hand V9 URDF.

Current live approach:

1. Read MANUS raw skeleton frames from `manus_raw_skeleton.csv`.
2. Use one open/neutral MANUS frame as the bend baseline.
3. Compute finger curl from MANUS bone-vector angles.
4. Drive only the V9 flexion joints and keep twist/spread quiet.

The old point-IK solver is still available, but it is not the default anymore.
For this hand it was too underconstrained: fingertips could be roughly right
while intermediate joints twisted into nonsense. The default live path is now
closer to the stable part of DexPilot / DEX-Retargeting style methods: use
vector geometry and constraints first, then add extra DOFs only after they are
calibrated.

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
  `i3 i2 i1 i0 m3 m2 m1 m0 r3 r2 r1 r0 p3 p2 p1 p0 t4 t3 t2 t1 t0`
- IK diagnostics:
  `ik_cost`, `ik_success`, `ik_nfev`, `mean_tip_error_m`

The sidecar `*.summary.json` contains aggregate error and calibration stats.

## Live MuJoCo Visualization

Run the native live viewer. This starts MANUS Core Integrated in WSL itself,
streams skeleton frames directly over stdout, retargets them, and writes the 21
Hand V9 joint positions straight into MuJoCo.

```powershell
cd C:\Users\henry\Desktop\hand_capture
python -m retargeting.live_manus_mujoco_hand_v8
```

Useful options:

```powershell
python -m retargeting.live_manus_mujoco_hand_v8 `
  --retarget-mode angle `
  --solve-every 1 `
  --finger-sign -1 `
  --thumb-sign 1 `
  --display-alpha 0.9 `
  --wrist-mode world
```

The live viewer defaults to `--retarget-mode angle`. That mode is fast and
curl-only; it should not randomly twist when you rotate your wrist. To compare
against the older point-IK path:

```powershell
python -m retargeting.live_manus_mujoco_hand_v8 --retarget-mode ik
```

If the four fingers curl backward, use `--finger-sign -1` or `--finger-sign 1`.
If only the thumb curls backward, use `--thumb-sign -1` or `--thumb-sign 1`.

The live script still records MANUS CSVs under:

```text
C:\Users\henry\Desktop\hand_capture\recordings\live_mujoco_...
```

## Joint Calibration

Run this when joints fold backward, rotate during pure finger curls, or need
runtime limits:

```powershell
cd C:\Users\henry\Desktop\hand_capture
python -m retargeting.calibrate_hand_v9
```

The script starts MANUS Integrated, then uses timed countdowns to guide you
through:

- neutral open hand
- full fist
- index/middle/ring/pinky isolated curls
- thumb curl
- finger spread

It writes:

```text
C:\Users\henry\Desktop\hand_capture\config\hand_v9_joint_calibration.json
```

The live MuJoCo viewer loads the new angle calibration automatically. It ignores
old IK-derived calibrations and stale angle profiles because those can corrupt
an otherwise stable curl solve. To test a calibration explicitly:

```powershell
python -m retargeting.live_manus_mujoco_hand_v8 `
  --joint-calibration C:\Users\henry\Desktop\hand_capture\config\hand_v9_joint_calibration.json
```

If you want manual Enter prompts instead of countdowns:

```powershell
python -m retargeting.calibrate_hand_v9 --manual-enter
```

Stop the dashboard session before running this standalone viewer; otherwise two
MANUS Core Integrated instances can fight over the same dongle. The MuJoCo
viewer uses `--wrist-mode world` by default so global wrist rotation is removed
before retargeting finger joints.

## Notes

- For the live viewer, hold an open/neutral hand for the first second so the
  angle baseline is sane.
- The thumb is the least trustworthy part of this first pass because the robot
  thumb has 5 DOF while the current MANUS keypoint selection has fewer directly
  comparable points.
- `--retarget-mode ik` is kept for experiments, but `angle` is the path to tune
  first.
- The angle mode intentionally drives the non-thumb base joints as fixed for
  now, but drives thumb `t0` because that joint provides important thumb travel
  in the V9 URDF.

## GeoRT Prep

GeoRT needs a human fingertip workspace and a robot fingertip workspace. Prepare
the local GeoRT checkout like this:

```powershell
cd C:\Users\henry\Desktop\hand_capture
.\.venv-geort\Scripts\python.exe -m retargeting.discover_v9_joint_limits
python -m retargeting.geort_prepare prepare-hand
python -m retargeting.geort_prepare export-human `
  --manus-csv C:\Users\henry\Desktop\hand_capture\recordings\live_mujoco_YYYY\manus_raw_skeleton.csv
```

`discover_v9_joint_limits` uses collision-mesh bounding boxes as conservative
proxies, sweeps each joint from neutral, and writes
`config\hand_v9_discovered_limits.json`. `prepare-hand` uses that file when it
generates `external\GeoRT\assets\hand_v9_right\robot_geort.urdf`.
