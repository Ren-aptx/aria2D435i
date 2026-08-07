#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


class Report:
    def __init__(self):
        self.items = []
        self.stats = {}
    def add(self, level, category, message):
        self.items.append((level, category, message))
    def ok(self, c, m): self.add('PASS', c, m)
    def warn(self, c, m): self.add('WARN', c, m)
    def fail(self, c, m): self.add('FAIL', c, m)
    def print(self):
        for level, cat, msg in self.items:
            print(f'[{level}] {cat}: {msg}')
        counts = Counter(x[0] for x in self.items)
        print(f"\nPASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
        if counts['FAIL']:
            print('Decision: NOT training-ready until FAIL items are fixed.')
        elif counts['WARN']:
            print('Decision: structurally usable; review WARN items and videos before training.')
        else:
            print('Decision: structural checks passed; run the training smoke test.')
        return counts


def finite_matrix(value: Any, shape):
    try:
        a = np.asarray(value, dtype=float)
    except Exception:
        return None
    return a if a.shape == shape and np.isfinite(a).all() else None


def check_se3(value, label, r: Report):
    T = finite_matrix(value, (4, 4))
    if T is None:
        r.fail('SE3', f'{label}: not a finite 4x4 matrix')
        return None
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-3):
        r.fail('SE3', f'{label}: invalid homogeneous bottom row')
    R = T[:3, :3]
    orth = np.linalg.norm(R.T @ R - np.eye(3))
    det = np.linalg.det(R)
    if orth > 5e-2 or abs(det - 1.0) > 5e-2:
        r.warn('SE3', f'{label}: rotation orth_err={orth:.4f}, det={det:.4f}')
    return T


def resolve_obs_path(raw: str, frame_dir: Path, session: Path):
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [
        frame_dir / p,
        session / p,
        session / 'preprocess' / p,
        session / 'preprocess' / 'all_data' / p,
    ]
    return next((x for x in candidates if x.is_file()), None)


def count_images(path: Path):
    if not path.is_dir(): return 0
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sum(p.is_file() and p.suffix.lower() in exts for p in path.iterdir())


def parse_trajectory(path: Path, r: Report, max_step: float):
    if not path.is_file():
        r.fail('trajectory', f'missing {path}')
        return
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        try: vals = [float(x) for x in line.replace(',', ' ').split()]
        except ValueError: continue
        if len(vals) >= 8 and all(math.isfinite(x) for x in vals[:8]): rows.append(vals)
    if not rows:
        r.fail('trajectory', 'no valid TUM-style pose rows')
        return
    r.ok('trajectory', f'{len(rows)} valid poses')
    ts = [x[0] for x in rows]
    bad = sum(b <= a for a, b in zip(ts, ts[1:]))
    (r.warn if bad else r.ok)('trajectory', f'timestamp non-increasing positions={bad}' if bad else 'timestamps strictly increasing')
    if len(rows) > 1:
        xyz = np.asarray([[x[1], x[2], x[3]] for x in rows])
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        large = int(np.sum(steps > max_step))
        r.stats['trajectory_steps_m'] = {'median': float(np.median(steps)), 'p95': float(np.percentile(steps,95)), 'max': float(np.max(steps))}
        (r.warn if large else r.ok)('trajectory', f'{large} steps exceed {max_step:.3f} m/frame; max={steps.max():.3f}' if large else f'no step exceeds {max_step:.3f} m/frame')


def parse_imu(path: Path, r: Report):
    if not path.is_file():
        r.warn('imu', f'missing {path}')
        return
    streams = defaultdict(list)
    with path.open(newline='', errors='replace') as f:
        for row in csv.reader(f):
            if len(row) < 5: continue
            name = row[1].strip().lower()
            if not (name.startswith('gyro') or name.startswith('accel')): continue
            try:
                ts = float(row[0]); xyz = list(map(float, row[2:5]))
            except ValueError: continue
            if math.isfinite(ts) and all(math.isfinite(v) for v in xyz):
                streams['gyro' if name.startswith('gyro') else 'accel'].append(ts)
    if not streams:
        r.fail('imu', 'no valid gyro/accel samples')
        return
    r.ok('imu', ', '.join(f'{k}={len(v)}' for k,v in streams.items()))
    if set(streams) != {'gyro','accel'}: r.fail('imu', 'both gyro and accel are required')
    for name, ts in streams.items():
        bad = sum(b <= a for a,b in zip(ts,ts[1:]))
        if bad: r.warn('imu', f'{name}: {bad} non-increasing timestamps')


def rotation_angle_deg(a, b):
    x = a.T @ b
    c = float((np.trace(x)-1)/2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def validate(session: Path, max_step: float, max_rot: float, r: Report):
    session = session.resolve()
    raw = session / 'raw'
    all_data = session / 'preprocess' / 'all_data'
    if not session.is_dir():
        r.fail('root', f'session not found: {session}'); return
    r.ok('root', str(session))

    aliases = {
        'rgb':['rgb','color'], 'depth':['depth'], 'aligned_depth':['aligned_depth','depth_aligned'],
        'ir_left':['ir_left','infrared_left','left_ir'], 'ir_right':['ir_right','infrared_right','right_ir']}
    counts = {}
    for name, names in aliases.items():
        folder = next((raw/x for x in names if (raw/x).is_dir()), None)
        counts[name] = count_images(folder) if folder else 0
    r.stats['raw_stream_counts'] = counts
    if any(counts.values()):
        r.ok('raw', ', '.join(f'{k}={v}' for k,v in counts.items()))
        nz = [v for v in counts.values() if v]
        if nz and (max(nz)-min(nz))/max(nz) > .02: r.warn('raw', 'stream counts differ by more than 2%')
    else: r.warn('raw', 'no recognized raw image directories')
    parse_trajectory(raw/'trajectory_rgb.txt', r, max_step)
    parse_imu(raw/'imu.csv', r)

    if not all_data.is_dir():
        r.fail('all_data', f'missing {all_data}'); return
    dirs = sorted(p for p in all_data.iterdir() if p.is_dir() and p.name.isdigit())
    if not dirs:
        r.fail('all_data', 'no numeric frame directories'); return
    r.ok('all_data', f'{len(dirs)} frame directories')
    idx = [int(p.name) for p in dirs]
    missing = sorted(set(range(min(idx),max(idx)+1))-set(idx))
    (r.warn if missing else r.ok)('all_data', f'{len(missing)} missing indices; first={missing[:10]}' if missing else 'indices continuous')

    training = [p/'training_data.json' for p in dirs if (p/'training_data.json').is_file()]
    if not training:
        r.fail('training_data', 'no training_data.json files'); return
    r.ok('training_data', f'{len(training)} files')

    poses, timestamps, hands, objects = [], [], Counter(), Counter()
    missing_paths, unreadable, empty_masks, full_masks = [], [], [], []
    parse_errors = 0
    for jf in training:
        try: d = json.loads(jf.read_text())
        except Exception as e:
            parse_errors += 1; r.fail('JSON', f'{jf}: {e}'); continue
        md, obs, ent = d.get('metadata'), d.get('obs'), d.get('entities')
        if not all(isinstance(x,dict) for x in (md,obs,ent)):
            r.fail('schema', f'{jf}: metadata/obs/entities missing'); continue
        K = finite_matrix(md.get('k'), (3,3))
        if K is None or K[0,0] <= 0 or K[1,1] <= 0: r.fail('schema', f'{jf}: invalid metadata.k')
        T = check_se3(md.get('c2w'), f'{jf}: metadata.c2w', r)
        try: ts = float(md.get('ts'))
        except Exception: ts = float('nan')
        if not math.isfinite(ts): r.fail('schema', f'{jf}: invalid metadata.ts')
        else:
            timestamps.append(ts)
            if T is not None: poses.append((ts,T,jf))
        for key,val in obs.items():
            if not (key.endswith('_path') and isinstance(val,str) and val): continue
            p = resolve_obs_path(val, jf.parent, session)
            if p is None: missing_paths.append(f'{jf}: {key}={val}'); continue
            if cv2 is not None:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None: unreadable.append(str(p)); continue
                if p.name.startswith('mask_'):
                    ratio = np.count_nonzero(img)/img.size
                    if ratio <= 1e-6: empty_masks.append(str(p))
                    if ratio >= .98: full_masks.append(str(p))
        for side, item in (ent.get('hands') or {}).items():
            if not isinstance(item,dict): continue
            hands[side] += 1
            key = next((k for k in item if k.startswith('T_') and k.endswith('_to_world')), None)
            if key: check_se3(item[key], f'{jf}: hand.{side}.{key}', r)
        for name, item in (ent.get('objects') or {}).items():
            if not isinstance(item,dict): continue
            objects[name] += 1
            key = next((k for k in item if k.startswith('T_') and k.endswith('_to_world')), None)
            if key: check_se3(item[key], f'{jf}: object.{name}.{key}', r)

    if parse_errors == 0: r.ok('JSON', 'all training JSON files parse')
    if missing_paths: r.fail('paths', f'{len(missing_paths)} referenced paths missing; first={missing_paths[:5]}')
    else: r.ok('paths', 'all referenced observation paths exist')
    if unreadable: r.fail('images', f'{len(unreadable)} images unreadable; first={unreadable[:5]}')
    elif cv2 is None: r.warn('images', 'opencv unavailable; decoding and mask checks skipped')
    else: r.ok('images', 'all referenced images decode')
    if empty_masks or full_masks: r.warn('masks', f'empty={len(empty_masks)}, nearly-full={len(full_masks)}')
    else: r.ok('masks', 'no empty/nearly-full referenced masks')
    if hands: r.ok('entities', 'hands: '+', '.join(f'{k}={v}' for k,v in hands.items()))
    else: r.fail('entities', 'no hands in training data')
    if objects: r.ok('entities', 'objects: '+', '.join(f'{k}={v}' for k,v in objects.items()))
    else: r.fail('entities', 'no objects in training data')

    if len(timestamps)>1:
        bad = sum(b<=a for a,b in zip(timestamps,timestamps[1:]))
        (r.fail if bad else r.ok)('timestamps', f'{bad} non-increasing training timestamps' if bad else 'training timestamps strictly increasing')
    poses.sort(key=lambda x:x[0])
    if len(poses)>1:
        tsteps=[]; rsteps=[]
        for (_,a,_),(_,b,_) in zip(poses,poses[1:]):
            tsteps.append(float(np.linalg.norm(b[:3,3]-a[:3,3])))
            rsteps.append(rotation_angle_deg(a[:3,:3],b[:3,:3]))
        lt=sum(x>max_step for x in tsteps); lr=sum(x>max_rot for x in rsteps)
        r.stats['training_pose_steps']={'translation_max_m':max(tsteps),'rotation_max_deg':max(rsteps)}
        (r.warn if lt else r.ok)('pose', f'{lt} translation jumps > {max_step} m; max={max(tsteps):.3f}' if lt else 'no excessive translation jumps')
        (r.warn if lr else r.ok)('pose', f'{lr} rotation jumps > {max_rot} deg; max={max(rsteps):.1f}' if lr else 'no excessive rotation jumps')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--session', required=True, type=Path)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--max-translation-step-m', type=float, default=.30)
    ap.add_argument('--max-rotation-step-deg', type=float, default=60.)
    ap.add_argument('--strict', action='store_true')
    a=ap.parse_args(); r=Report()
    validate(a.session,a.max_translation_step_m,a.max_rotation_step_deg,r)
    counts=r.print()
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps({'session':str(a.session.resolve()),'summary':dict(counts),'stats':r.stats,'findings':[{'level':x[0],'category':x[1],'message':x[2]} for x in r.items]},indent=2,ensure_ascii=False))
        print(f'Report written: {a.report}')
    if counts['FAIL']: return 2
    if a.strict and counts['WARN']: return 1
    return 0

if __name__=='__main__': sys.exit(main())
