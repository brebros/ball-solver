#!/usr/bin/env python3
"""
ball_solver_agent.py — single-file multi-rail bank/kick solver (billiards / 8-ball).

=== SPEC FOR AI AGENTS & CALLERS ===

PURPOSE
    Given a table size and three points (cue ball, object ball, pocket), find the
    path that uses 0..N cushion bounces and rank the solutions by difficulty.

COORDINATE SYSTEM
    Table = rectangle [0,W] x [0,H]. (0,0) = one corner, (W,H) = the opposite corner.
    Units are arbitrary as long as x and y share the same unit. Cushions:
        L = x=0,  R = x=W,  B = y=0,  T = y=H
    Standard pockets: (0,0) (W,0) (0,H) (W,H) (0,H/2) (W,H/2).

MODES
    kick : cue ball bounces off cushion(s) FIRST, then contacts the object ball.
    bank : object ball bounces off cushion(s), then drops into the pocket.

PUBLIC API
    solve_kick(C, O, P, W, H, max_rails=3) -> list[Solution]
    solve_bank(O, P, C, W, H, max_rails=3) -> list[Solution]
    Solution = {rails, pts, len, cut, aim, score [, ghost]}
        rails : cushion codes in the order hit, e.g. ['L','T']
        pts   : full geometric path [(x,y), ...] incl. start and end
        len   : total path length (same unit as coords)
        cut   : cut angle in degrees; 0 = straight / full-ball hit
        aim   : aim direction in degrees, atan2(dy,dx), 0 = +x axis
        score : difficulty score, LOWER is easier.  rails*100 + len*10 + cut*2
    Candidates with cut >= 85 deg are discarded as physically impossible.

CLI
    python ball_solver_agent.py                        # interactive prompts
    python ball_solver_agent.py --mode kick --cue 0.9 0.25 \
        --object 1.6 0.95 --pocket 2.24 1.12 --table 2.24 1.12 --max-rails 3 --json
    python ball_solver_agent.py --self-test            # built-in sanity checks

JSON OUTPUT (--json)
    {"ok": true, "mode": "kick", "table": [W,H], "max_rails": 3,
     "solutions": [ {rails, pts, len, cut, aim, score}, ... ]}

NOTES
    - Pure geometry (unfolding / mirror method). No hooks, no game client access.
    - Every returned path is re-checked against the law of reflection before output.
    - Python 3.8+, standard library only.
"""

import argparse
import itertools
import json
import math
import sys

RAILS = ('L', 'R', 'B', 'T')
BALL_R = 0.0286            # ball radius, for the ghost-ball offset (any consistent unit)


# ---------------------------------------------------------------- small vector math
def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _mul(a, s): return (a[0] * s, a[1] * s)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1]
def _dist(a, b): return math.hypot(b[0] - a[0], b[1] - a[1])


def _norm(v):
    d = math.hypot(v[0], v[1])
    return (v[0] / d, v[1] / d) if d else (0.0, 0.0)


def _angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(_norm(a), _norm(b))))))


def _aim(v):
    return math.degrees(math.atan2(v[1], v[0])) % 360


# ---------------------------------------------------------------- rails & reflection
def _reflect(p, rail, W, H):
    x, y = p
    return {'L': (-x, y), 'R': (2 * W - x, y),
            'B': (x, -y), 'T': (x, 2 * H - y)}[rail]


def _rail_geom(rail, W, H):
    return {'L': ('v', 0.0), 'R': ('v', W), 'B': ('h', 0.0), 'T': ('h', H)}[rail]


def _cross(p0, p1, rail, W, H):
    kind, c = _rail_geom(rail, W, H)
    x0, y0 = p0
    x1, y1 = p1
    if kind == 'v':
        if x1 == x0:
            return None, None
        t = (c - x0) / (x1 - x0)
        return (c, y0 + t * (y1 - y0)), t
    if y1 == y0:
        return None, None
    t = (c - y0) / (y1 - y0)
    return (x0 + t * (x1 - x0), c), t


def _on_rail(p, rail, W, H, eps=1e-6):
    kind, _ = _rail_geom(rail, W, H)
    return (-eps <= p[1] <= H + eps) if kind == 'v' else (-eps <= p[0] <= W + eps)


def _build(S, T, rails, W, H):
    """Unfold S->T over the given cushion sequence. Returns (pts, length) or None."""
    end = T
    for r in reversed(rails):
        end = _reflect(end, r, W, H)
    start, pts, length = S, [S], 0.0
    for r in rails:
        B, t = _cross(start, end, r, W, H)
        if B is None or not (0.0 <= t <= 1.0) or not _on_rail(B, r, W, H):
            return None
        pts.append(B)
        length += _dist(start, B)
        end = _reflect(end, r, W, H)
        start = B
    pts.append(T)
    length += _dist(start, T)
    return pts, length


def _verify(pts, rails, W, H, tol=1e-3):
    """Law-of-reflection check at every bounce; guards against numeric junk."""
    for i, r in enumerate(rails):
        A, Bp, C = pts[i], pts[i + 1], pts[i + 2]
        inc, out = _norm(_sub(Bp, A)), _norm(_sub(C, Bp))
        kind, _ = _rail_geom(r, W, H)
        if kind == 'v':
            if abs(out[0] + inc[0]) > tol or abs(out[1] - inc[1]) > tol:
                return False
        else:
            if abs(out[1] + inc[1]) > tol or abs(out[0] - inc[0]) > tol:
                return False
    return True


def _solve(S, T, W, H, max_rails=3):
    """All S->T paths with 0..max_rails bounces, verified. Unsorted."""
    out = []
    for n in range(max_rails + 1):
        for rails in itertools.product(RAILS, repeat=n):
            if any(rails[i] == rails[i + 1] for i in range(n - 1)):
                continue
            res = _build(S, T, list(rails), W, H)
            if res is None:
                continue
            pts, length = res
            if not _verify(pts, list(rails), W, H):
                continue
            out.append({'rails': list(rails), 'pts': pts, 'len': length})
    return out


# ---------------------------------------------------------------- public API
def solve_kick(C, O, P, W, H, max_rails=3):
    """Cue C -> (cushion xN) -> object O, so that O heads toward pocket P."""
    d_obj = _norm(_sub(P, O))
    ghost = _sub(O, _mul(d_obj, 2 * BALL_R))
    res = []
    for c in _solve(C, ghost, W, H, max_rails):
        pts = c['pts']
        d_cue = _norm(_sub(pts[-1], pts[-2])) if len(pts) >= 2 else _norm(_sub(ghost, C))
        cut = _angle(d_cue, d_obj)
        if cut >= 85:
            continue
        res.append({'rails': c['rails'], 'pts': c['pts'], 'len': round(c['len'], 4),
                    'cut': round(cut, 2), 'aim': round(_aim(d_cue), 2),
                    'score': round(len(c['rails']) * 100 + c['len'] * 10 + cut * 2, 2)})
    res.sort(key=lambda x: x['score'])
    return res


def solve_bank(O, P, C, W, H, max_rails=3):
    """Object O -> (cushion xN) -> pocket P. Cue is given for cut-angle / aim."""
    res = []
    for c in _solve(O, P, W, H, max_rails):
        pts = c['pts']
        d_obj = _norm(_sub(pts[1], pts[0])) if len(pts) >= 2 else _norm(_sub(P, O))
        ghost = _sub(O, _mul(d_obj, 2 * BALL_R))
        d_cue = _norm(_sub(ghost, C))
        cut = _angle(d_cue, d_obj)
        if cut >= 85:
            continue
        res.append({'rails': c['rails'], 'pts': c['pts'], 'len': round(c['len'], 4),
                    'cut': round(cut, 2), 'aim': round(_aim(d_cue), 2),
                    'ghost': (round(ghost[0], 4), round(ghost[1], 4)),
                    'score': round(len(c['rails']) * 100 + c['len'] * 10 + cut * 2, 2)})
    res.sort(key=lambda x: x['score'])
    return res


# ---------------------------------------------------------------- self test
def _self_test():
    W, H = 2.24, 1.12
    ks = solve_kick((0.90, 0.25), (1.60, 0.95), (2.24, 1.12), W, H)
    bs = solve_bank((0.60, 0.40), (0.05, 0.05), (1.80, 0.90), W, H)
    assert ks, "kick returned no solutions"
    assert bs, "bank returned no solutions"
    # a synthetic 2-rail path must be found and must satisfy the reflection law
    tp = _build((1.5, 0.4), (1.08, 0.7467), ['B', 'L'], W, H)
    assert tp and _verify(tp[0], ['B', 'L'], W, H), "2-rail verification failed"
    print(f"self-test OK: kick={len(ks)} sols, bank={len(bs)} sols, verification pass")
    return True


# ---------------------------------------------------------------- CLI
def _batch(mode, C, O, P, W, H, max_rails, as_json):
    if mode == 'kick':
        sols = solve_kick(C, O, P, W, H, max_rails)
    else:
        sols = solve_bank(O, P, C, W, H, max_rails)
    if as_json:
        print(json.dumps({"ok": True, "mode": mode, "table": [W, H],
                          "max_rails": max_rails, "solutions": sols}))
        return 0
    if not sols:
        print("no valid path found")
        return 1
    for s in sols:
        tag = ' -> '.join(s['rails']) or 'direct'
        print(f"[{tag:12}] aim={s['aim']:6.1f}deg  cut={s['cut']:5.1f}deg  "
              f"len={s['len']:.3f}  score={s['score']:.1f}")
        print("   path: " + '  '.join(f"({p[0]:.2f},{p[1]:.2f})" for p in s['pts']))
    return 0


def _interactive(W, H):
    print(f"table {W} x {H}  |  coords: x in 0..{W}, y in 0..{H}")
    while True:
        try:
            mode = input("mode [kick/bank/quit]: ").strip().lower()
        except EOFError:
            break
        if mode in ('quit', 'q', ''):
            break
        if mode not in ('kick', 'bank'):
            print("  unknown mode")
            continue
        try:
            cue = tuple(map(float, input("  cue    (x y): ").replace(',', ' ').split()))
            obj = tuple(map(float, input("  object (x y): ").replace(',', ' ').split()))
            pkt = tuple(map(float, input("  pocket (x y): ").replace(',', ' ').split()))
        except Exception as e:
            print("  bad input:", e)
            continue
        sols = (_batch_sols(mode, cue, obj, pkt, W, H, 3))
        if not sols:
            print("  no valid path")
            continue
        for s in sols[:3]:
            tag = ' -> '.join(s['rails']) or 'direct'
            print(f"  [{tag:10}] aim={s['aim']:6.1f}deg  cut={s['cut']:5.1f}deg  len={s['len']:.3f}")
    return 0


def _batch_sols(mode, cue, obj, pkt, W, H, max_rails):
    return (solve_kick(cue, obj, pkt, W, H, max_rails) if mode == 'kick'
            else solve_bank(obj, pkt, cue, W, H, max_rails))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-rail bank/kick solver (billiards/8-ball)")
    ap.add_argument('--mode', choices=['kick', 'bank'])
    ap.add_argument('--cue', nargs=2, type=float, metavar=('X', 'Y'))
    ap.add_argument('--object', nargs=2, type=float, metavar=('X', 'Y'))
    ap.add_argument('--pocket', nargs=2, type=float, metavar=('X', 'Y'))
    ap.add_argument('--table', nargs=2, type=float, default=[2.24, 1.12], metavar=('W', 'H'))
    ap.add_argument('--max-rails', type=int, default=3)
    ap.add_argument('--json', action='store_true', dest='as_json')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    W, H = args.table
    have_all = args.mode and args.cue and args.object and args.pocket
    have_none = not args.mode and not args.cue and not args.object and not args.pocket
    if have_all:
        return _batch(args.mode, tuple(args.cue), tuple(args.object), tuple(args.pocket),
                      W, H, args.max_rails, args.as_json)
    if have_none:
        return _interactive(W, H)
    print("error: give --mode + --cue + --object + --pocket together, or no args for interactive",
          file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
