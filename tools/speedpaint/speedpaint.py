#!/usr/bin/env python3
"""Turn a finished drawing into a WIP / speedpaint-style timelapse video.

The video rebuilds the piece the way a drawing usually comes together:

   1. construction  - line of action, rough silhouette shapes and ovals
   2. sketch        - thin, tapered pencil strokes with grain, some re-stated
   3. fixes         - a few wrong strokes erased and redrawn
   4. transform     - the sketch is selected and nudged/scaled into place
   5. inking        - the real line art traced over the lowered sketch
   6. base colours  - each area filled in piece by piece, spreading from the brush
   7. colour adjust - hue/saturation tweak from first-pass colours to final ones
   8. shading       - shadows painted in area by area
   9. rendering     - highlights and detail up to the finished piece
  10. background    - optional background version swept in, then a hold

Usage:
    python3 speedpaint.py final.png -o speedpaint.mp4
    python3 speedpaint.py final.png --lineart lines.png --bg-final final_bg.png -o wip.mp4
    python3 speedpaint.py final.png --duration 90 --no-cursor --frames-dir stills

--lineart should be your line art layer exported on white, the same size as the
final image. Without it the lines are estimated from the final image.

Requires: numpy, opencv-python-headless, scikit-image, ffmpeg on PATH.
"""

import argparse
import colorsys
import os
import subprocess
import sys

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize


# =========================================================================== #
# Loading
# =========================================================================== #

def load_image(path, max_side=None, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # flatten transparency onto white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    if size is not None:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nh, nw = int(h * scale) // 2 * 2, int(w * scale) // 2 * 2  # even dims for h264
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def estimate_line_mask(img):
    """Fallback when no line art layer is given."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12)
    dark &= (gray < 140).astype(np.uint8) * 255
    edges = cv2.Canny(cv2.bilateralFilter(gray, 7, 50, 50), 60, 150)
    mask = cv2.bitwise_or(dark, edges)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    return (keep[labels] * 255).astype(np.uint8)


# =========================================================================== #
# Path geometry
# =========================================================================== #

def trace_skeleton(mask, min_pts=4):
    """Skeletonise a line mask and trace it into ordered polylines (x, y)."""
    sk = skeletonize(mask > 0)
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        y, x = p
        out = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (y + dy, x + dx) in pix:
                out.append((y + dy, x + dx))
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (y + dy, x + dx)
            # Skip diagonals already reachable through an orthogonal step,
            # so staircases don't look like junctions.
            if q in pix and (y + dy, x) not in pix and (y, x + dx) not in pix:
                out.append(q)
        return out

    nb = {p: nbrs(p) for p in pix}
    used = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        used.add((a, b)); used.add((b, a))
        prev, cur = a, b
        while len(nb[cur]) == 2:
            nxt = [q for q in nb[cur] if q != prev and (cur, q) not in used]
            if not nxt:
                break
            q = nxt[0]
            used.add((cur, q)); used.add((q, cur))
            path.append(q)
            prev, cur = cur, q
        return path

    for p in pix:
        if len(nb[p]) != 2:
            for q in nb[p]:
                if (p, q) not in used:
                    paths.append(walk(p, q))
    for p in pix:  # closed loops
        if len(nb[p]) == 2 and (p, nb[p][0]) not in used:
            paths.append(walk(p, nb[p][0]))
    return [np.array([(x, y) for y, x in pth], np.float32) for pth in paths if len(pth) >= min_pts]


def resample(pts, step=1.5):
    if len(pts) < 2:
        return pts
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] < step:
        return pts
    t = np.arange(0, d[-1] + 1e-6, step)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1).astype(np.float32)


def smooth(pts, k=5):
    if len(pts) < k + 2:
        return pts
    pad = np.pad(pts, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(2)], 1).astype(np.float32)


def jitter(pts, amp, rng, wavelength=35.0):
    """Low-frequency wobble, like a hand that isn't perfectly steady."""
    if len(pts) < 3 or amp <= 0:
        return pts
    n = max(2, int(len(pts) / wavelength) + 2)
    offs = rng.normal(0, amp, (n, 2))
    idx = np.linspace(0, n - 1, len(pts))
    j = np.stack([np.interp(idx, np.arange(n), offs[:, i]) for i in range(2)], 1)
    return (pts + j).astype(np.float32)


def overshoot(pts, rng, lo, hi):
    """Extend both ends a little along their tangent - typical of loose sketching."""
    if len(pts) < 4:
        return pts
    head, tail = pts[:1], pts[-1:]
    parts = [pts]
    for end, inner, front in ((pts[0], pts[min(3, len(pts) - 1)], True),
                              (pts[-1], pts[max(-4, -len(pts))], False)):
        d = end - inner
        nrm = np.linalg.norm(d)
        if nrm < 1e-3:
            continue
        ext = rng.uniform(lo, hi)
        steps = np.linspace(0, ext, max(2, int(ext / 1.5)))[1:, None]
        extra = end + d / nrm * steps
        parts = [extra[::-1]] + parts if front else parts + [extra]
    return np.concatenate(parts).astype(np.float32)


def split(paths, lo, hi, rng, step=1.5):
    """Cut long paths into stroke-sized pieces (lengths in px)."""
    out = []
    for p in paths:
        i = 0
        while i < len(p) - 2:
            n = max(2, int(rng.uniform(lo, hi) / step))
            out.append(p[i:i + n + 1])
            i += n
    return [s for s in out if len(s) >= 3]


def order_nn(strokes):
    """Greedy nearest-neighbour order so the 'pen' moves like a person's would."""
    if not strokes:
        return []
    starts = np.array([s[0] for s in strokes])
    ends = np.array([s[-1] for s in strokes])
    used = np.zeros(len(strokes), bool)
    i = int(np.argmin(starts[:, 1] * 3 + starts[:, 0]))
    order = [strokes[i]]
    used[i] = True
    cur = strokes[i][-1]
    for _ in range(len(strokes) - 1):
        d0 = np.sum((starts - cur) ** 2, 1)
        d1 = np.sum((ends - cur) ** 2, 1)
        d0[used] = np.inf
        d1[used] = np.inf
        a, b = int(np.argmin(d0)), int(np.argmin(d1))
        if d0[a] <= d1[b]:
            s = strokes[a]; used[a] = True
        else:
            s = strokes[b][::-1].copy(); used[b] = True
        order.append(s)
        cur = s[-1]
    return order


def apply_affine(pts, M):
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


# =========================================================================== #
# Brushes
# =========================================================================== #

def pressure(n, i0, i1):
    idx = np.arange(i0, i1 + 1)
    p = np.sin(np.pi * np.clip((idx + 0.5) / max(1, n), 0, 1)) ** 0.6
    return 0.2 + 0.8 * p


def bbox(seg, pad, w, h):
    x0 = max(0, int(seg[:, 0].min()) - pad); y0 = max(0, int(seg[:, 1].min()) - pad)
    x1 = min(w, int(seg[:, 0].max()) + pad + 1); y1 = min(h, int(seg[:, 1].max()) + pad + 1)
    return x0, y0, x1, y1


def pencil(layer, grain, s, i0, i1, radius, opacity):
    """Tapered, grainy pencil line; opacity builds up where strokes overlap."""
    h, w = layer.shape
    seg = s.pts[i0:i1 + 1]
    if len(seg) < 2:
        return
    pr = pressure(len(s.pts), i0, i1) * s.weight
    pad = int(radius * 2) + 3
    x0, y0, x1, y1 = bbox(seg, pad, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.round((seg - [x0, y0]) * 16).astype(np.int32)
    for k in range(len(seg) - 1):
        th = max(1, int(round(2 * radius * pr[k])))
        val = int(255 * min(1.0, opacity * pr[k]))
        cv2.line(tmp, tuple(int(v) for v in q[k]), tuple(int(v) for v in q[k + 1]), val, th, cv2.LINE_AA, 4)
    a = tmp.astype(np.float32) / 255.0 * grain[y0:y1, x0:x1]
    L = layer[y0:y1, x0:x1]
    L += a * (1 - L)


def stamp_line(mask, seg, thickness, value=255):
    if len(seg) < 2:
        return
    q = np.round(seg * 16).astype(np.int32)
    cv2.polylines(mask, [q], False, value, max(1, int(thickness)), cv2.LINE_AA, 4)


class Stroke:
    __slots__ = ("pts", "weight", "kind", "data")

    def __init__(self, pts, weight=1.0, kind="draw", data=None):
        self.pts, self.weight, self.kind, self.data = pts, weight, kind, data


def play(strokes, nfr, draw, frame_fn):
    """Spread the drawing of `strokes` over `nfr` frames; draw(s, i0, i1)."""
    total = sum(len(s.pts) - 1 for s in strokes)
    per = total / max(1, nfr)
    si, pi, carry = 0, 0, 0.0
    for f in range(nfr):
        carry += per
        if f == nfr - 1:
            carry = float("inf")  # make sure everything is drawn
        last, kind = None, "draw"
        while carry >= 1 and si < len(strokes):
            s = strokes[si]
            take = min(len(s.pts) - 1 - pi, carry)
            take = int(take)
            if take <= 0:
                si, pi = si + 1, 0
                continue
            draw(s, pi, pi + take)
            pi += take
            carry -= take
            last, kind = s.pts[pi], s.kind
            if pi >= len(s.pts) - 1:
                si, pi = si + 1, 0
        frame_fn(last, kind)


# =========================================================================== #
# Compositing / overlays
# =========================================================================== #

def blend(base, top, alpha):
    a = alpha[:, :, None] if alpha.ndim == 2 else alpha
    return (base.astype(np.float32) * (1 - a) + top.astype(np.float32) * a).astype(np.uint8)


def draw_cursor(frame, pt, radius):
    if pt is None:
        return frame
    c = (int(pt[0]), int(pt[1]))
    cv2.circle(frame, c, int(radius), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, c, int(radius), (30, 30, 30), 1, cv2.LINE_AA)
    return frame


def dashed_rect(frame, quad, t):
    """Transform box: dashed outline and handles."""
    pts = np.round(quad).astype(int)
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        L = np.linalg.norm(b - a)
        n = max(1, int(L / 8))
        for k in range(n):
            if (k + int(t)) % 2:
                continue
            p = a + (b - a) * k / n
            q = a + (b - a) * min(1, (k + 1) / n)
            cv2.line(frame, tuple(int(v) for v in p), tuple(int(v) for v in q), (40, 40, 40), 1, cv2.LINE_AA)
    handles = list(pts) + [(pts[i] + pts[(i + 1) % 4]) // 2 for i in range(4)]
    for p in handles:
        x, y = int(p[0]), int(p[1])
        cv2.rectangle(frame, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), -1)
        cv2.rectangle(frame, (x - 4, y - 4), (x + 4, y + 4), (60, 60, 60), 1)
    return frame


def hsl_panel(frame, hue, sat, light):
    """Small Hue/Saturation dialog in the corner while colours are adjusted."""
    h, w = frame.shape[:2]
    pw, ph = 210, 104
    x0, y0 = w - pw - 14, 14
    roi = frame[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi.astype(np.float32) * 0.15 + np.array([48, 44, 42]) * 0.85).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), 1)
    cv2.putText(frame, "Hue / Saturation", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)
    for i, (name, v) in enumerate((("Hue", hue), ("Sat", sat), ("Light", light))):
        y = y0 + 40 + i * 24
        cv2.putText(frame, name, (x0 + 10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
        sx0, sx1 = x0 + 58, x0 + pw - 44
        cv2.line(frame, (sx0, y), (sx1, y), (120, 120, 120), 2)
        kx = int(sx0 + (sx1 - sx0) * (0.5 + np.clip(v, -50, 50) / 100))
        cv2.circle(frame, (kx, y), 5, (235, 235, 235), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(round(v)):+d}", (sx1 + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return frame


class Writer:
    def __init__(self, path, w, h, fps, frames_dir=None):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-preset", "medium", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.count = 0
        self.frames_dir = frames_dir
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)

    def write(self, frame, still=None):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if self.frames_dir and still:
            cv2.imwrite(os.path.join(self.frames_dir, f"{self.count:05d}_{still}.png"), frame)
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# =========================================================================== #
# Colour analysis
# =========================================================================== #

def materials(img_nolines, fg, k, rng):
    """Cluster the piece into 'materials' (hair, jacket, skin...) by hue rather
    than shade, so each gets a single base colour like a real flat."""
    lab = cv2.cvtColor(img_nolines, cv2.COLOR_BGR2LAB).astype(np.float32)
    feat = np.stack([lab[:, :, 1], lab[:, :, 2], lab[:, :, 0] * 0.35], 2)
    pts = feat[fg]
    sample = pts[rng.choice(len(pts), min(len(pts), 60000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centres = cv2.kmeans(sample, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    idx = np.empty(len(pts), np.uint8)
    for s in range(0, len(pts), 200000):
        d = ((pts[s:s + 200000, None, :] - centres[None]) ** 2).sum(2)
        idx[s:s + 200000] = d.argmin(1)
    mat = np.full(fg.shape, 255, np.uint8)
    mat[fg] = idx
    mat = cv2.medianBlur(mat, 7)
    mat[~fg] = 255
    base = np.zeros((k, 3), np.uint8)
    L = lab[:, :, 0]
    for c in range(k):
        m = mat == c
        if m.sum() < 20:
            continue
        lv = L[m]
        lo, hi = np.percentile(lv, 60), np.percentile(lv, 85)  # lit side, ignoring shadows
        sel = m & (L >= lo) & (L <= hi)
        base[c] = np.median(img_nolines[sel if sel.any() else m], axis=0)
    return mat, base


def shift_colour(bgr, rng):
    b, g, r = (int(v) / 255 for v in bgr)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    hh = (hh + rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)) % 1
    ss = float(np.clip(ss * rng.uniform(0.7, 0.9), 0, 1))
    ll = float(np.clip(ll + rng.uniform(-0.04, 0.06), 0, 1))
    r, g, b = colorsys.hls_to_rgb(hh, ll, ss)
    return np.array([b * 255, g * 255, r * 255], np.uint8)


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="the finished piece")
    ap.add_argument("-o", "--output", default="speedpaint.mp4")
    ap.add_argument("--lineart", help="line art layer exported on white, same size as the image")
    ap.add_argument("--bg-final", help="a version with background, shown as the last step")
    ap.add_argument("--duration", type=float, default=60, help="target length in seconds (default 60)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=1280, help="longest side of the video in px")
    ap.add_argument("--materials", type=int, default=9, help="number of base colours to block in")
    ap.add_argument("--sketch-colour", default="70,80,115", help="sketch pencil colour R,G,B")
    ap.add_argument("--no-cursor", action="store_true")
    ap.add_argument("--frames-dir", help="also save a still at the end of each stage")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img = load_image(args.image, args.max_side)
    h, w = img.shape[:2]
    sc = min(w, h) / 800.0  # size scale for brushes
    paper = np.full_like(img, 255)
    show_cursor = not args.no_cursor

    print(f"Canvas {w}x{h}; analysing...")
    # ---------------- line art ---------------- #
    if args.lineart:
        la = load_image(args.lineart, size=(w, h))
        la_gray = cv2.cvtColor(la, cv2.COLOR_BGR2GRAY).astype(np.float32)
        line_alpha = np.clip((255 - la_gray) / 200.0, 0, 1)
        a3 = np.maximum(line_alpha, 0.05)[:, :, None]
        line_col = np.clip(255 - (255 - la.astype(np.float32)) / a3, 0, 255).astype(np.uint8)
        lines = (line_alpha > 0.12).astype(np.uint8) * 255
    else:
        lines = estimate_line_mask(img)
        line_col = (img.astype(np.float32) * 0.8).astype(np.uint8)
        line_alpha = cv2.GaussianBlur(lines, (3, 3), 0).astype(np.float32) / 255.0

    def with_lines(base):
        return blend(base, line_col, line_alpha)

    skel = [resample(p, 1.5) for p in trace_skeleton(lines)]
    skel = [p for p in skel if len(p) >= 4]

    # ---------------- silhouette / materials ---------------- #
    no_lines = cv2.inpaint(img, cv2.dilate(lines, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    fg = np.abs(no_lines.astype(np.int16) - 255).sum(2) > 30
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mat_idx, base_cols = materials(no_lines, fg, args.materials, rng)
    flats = paper.copy()
    for c in range(len(base_cols)):
        flats[mat_idx == c] = base_cols[c]
    wrong_cols = np.array([shift_colour(c, rng) for c in base_cols])
    flats_wrong = paper.copy()
    for c in range(len(base_cols)):
        flats_wrong[mat_idx == c] = wrong_cols[c]
    ksz = int(25 * sc) | 1
    sil = cv2.morphologyEx(((fg | (lines > 0)) * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))

    # ---------------- timeline ---------------- #
    share = dict(construct=0.07, sketch=0.17, fixes=0.05, transform=0.035, ink=0.20, block=0.12,
                 colour=0.04, shade=0.12, render=0.10, bg=0.05 if args.bg_final else 0.0, hold=0.04)
    norm = sum(share.values())
    total = int(args.duration * args.fps)
    nfr = {k: int(round(total * v / norm)) for k, v in share.items()}

    out = Writer(args.output, w, h, args.fps, args.frames_dir)
    sk_rgb = np.array([int(v) for v in args.sketch_colour.split(",")][::-1], np.uint8)
    cons_rgb = np.array([235, 190, 140], np.uint8)  # light blue (BGR)
    grain = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 0.7)
    grain = 0.45 + 0.55 * (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    sk_full = np.broadcast_to(sk_rgb, img.shape)
    cons_full = np.broadcast_to(cons_rgb, img.shape)

    cons = np.zeros((h, w), np.float32)     # construction layer
    sketch = np.zeros((h, w), np.float32)   # sketch layer
    pen_r, eraser_r = 3 * sc + 2, 9 * sc + 3

    def emit(frame, pt=None, r=None, still=None):
        if show_cursor and pt is not None:
            frame = draw_cursor(frame, pt, r or pen_r)
        out.write(frame, still)

    def sketch_frame(cons_op=1.0, sk_layer=None):
        f = blend(paper, cons_full, cons * cons_op)
        return blend(f, sk_full, sketch if sk_layer is None else sk_layer)

    # ---------------- 1. construction ---------------- #
    print("1 construction")
    c_paths = []
    ys_fg = np.nonzero(sil.any(1))[0]
    bands = np.linspace(ys_fg.min(), ys_fg.max(), 12)
    spine = []
    for y0b, y1b in zip(bands[:-1], bands[1:]):
        _, xx = np.nonzero(sil[int(y0b):int(y1b) + 1])
        if len(xx):
            spine.append((xx.mean(), (y0b + y1b) / 2))
    if len(spine) > 2:
        sp = smooth(resample(np.array(spine, np.float32), 1.5), 25)
        c_paths.append(jitter(sp, 3 * sc, rng, 80))
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        poly = cv2.approxPolyDP(c, 0.012 * cv2.arcLength(c, True), True)[:, 0, :].astype(np.float32)
        for i in range(len(poly)):
            seg = resample(np.array([poly[i], poly[(i + 1) % len(poly)]]), 1.5)
            if len(seg) > 6:
                c_paths.append(overshoot(jitter(seg, 2 * sc, rng), rng, 4 * sc, 14 * sc))
    for c in range(len(base_cols)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mat_idx == c).astype(np.uint8))
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < 0.006 * w * h:
                continue
            cc, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            big = max(cc, key=len)
            if len(big) < 5:
                continue
            (ex, ey), (ea, eb), ang = cv2.fitEllipse(big)
            ell = cv2.ellipse2Poly((int(ex), int(ey)), (max(1, int(ea / 2 * 0.9)), max(1, int(eb / 2 * 0.9))),
                                   int(ang), 0, 360, 4)
            ell = resample(ell.astype(np.float32), 1.5)
            half = len(ell) // 2 + int(len(ell) * 0.08)
            for part in (ell[:half], ell[len(ell) // 2:]):
                if len(part) > 4:
                    c_paths.append(jitter(part, 2.5 * sc, rng, 50))
    c_strokes = [Stroke(p, rng.uniform(0.7, 1.0)) for p in order_nn(c_paths)]
    play(c_strokes, nfr["construct"],
         lambda s, a, b: pencil(cons, grain, s, a, b, 1.3 * sc + 0.4, 0.55),
         lambda pt, k: emit(sketch_frame(), pt))
    emit(sketch_frame(), still="construction")

    # ---------------- 2. sketch (drawn slightly off, fixed later) ---------------- #
    print("2 sketch")
    cy, cx = np.argwhere(sil > 0).mean(0)
    M_wrong = cv2.getRotationMatrix2D((float(cx), float(cy)), -1.6, 1.035)
    M_wrong[:, 2] += (9 * sc, -7 * sc)
    pieces = split([smooth(p, 4) for p in skel], 18 * sc, 70 * sc, rng)
    sk_paths = []
    for p in pieces:
        p = apply_affine(overshoot(jitter(p, 1.6 * sc, rng), rng, 1.5 * sc, 7 * sc), M_wrong)
        sk_paths.append(p)
        if rng.random() < 0.35:  # a restated "searching" line next to it
            sk_paths.append(jitter(p, 1.8 * sc, rng, 25))
    sk_paths = order_nn(sk_paths)
    # A few strokes go wrong first; they get erased and redrawn in the fixes stage.
    long_ids = [i for i, s in enumerate(sk_paths) if len(s) > 30 * sc and i < 0.8 * len(sk_paths)]
    bad_ids = sorted(rng.choice(long_ids, min(len(long_ids), 8), replace=False).tolist()) if long_ids else []
    bad = {}
    for i in bad_ids:
        s = sk_paths[i]
        d = s[-1] - s[0]
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        sign = rng.choice([-1, 1])
        bend = np.sin(np.linspace(0, np.pi, len(s)))[:, None] * nrm * sign * rng.uniform(10, 18) * sc
        bad[i] = (s + bend + nrm * sign * 3 * sc).astype(np.float32)
    strokes = [Stroke(bad.get(i, s), rng.uniform(0.75, 1.0)) for i, s in enumerate(sk_paths)]
    cons_fade = {"f": 0}

    def sketch_tick(pt, k):
        t = cons_fade["f"] / max(1, nfr["sketch"] - 1)
        cons_fade["f"] += 1
        emit(sketch_frame(cons_op=1.0 - 0.5 * t), pt)

    play(strokes, nfr["sketch"], lambda s, a, b: pencil(sketch, grain, s, a, b, 0.9 * sc + 0.3, 0.75), sketch_tick)
    emit(sketch_frame(cons_op=0.5), still="sketch")

    # ---------------- 3. fixes: erase and redraw ---------------- #
    print("3 fixes")
    fix_strokes = []
    for i in bad_ids:
        fix_strokes.append(Stroke(bad[i], 1.0, "erase"))
        fix_strokes.append(Stroke(sk_paths[i], 1.0, "draw"))

    def fix_draw(s, a, b):
        seg = s.pts[a:b + 1]
        if s.kind == "erase":
            x0, y0, x1, y1 = bbox(seg, int(eraser_r) + 2, w, h)
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            stamp_line(m, seg - [x0, y0], 2 * eraser_r)
            sketch[y0:y1, x0:x1] *= 1 - m.astype(np.float32) / 255.0
        else:
            pencil(sketch, grain, s, a, b, 0.9 * sc + 0.3, 0.8)

    play(fix_strokes, nfr["fixes"], fix_draw,
         lambda pt, k: emit(sketch_frame(cons_op=0.5), pt, eraser_r if k == "erase" else pen_r))
    emit(sketch_frame(cons_op=0.5), still="fixes")

    # ---------------- 4. transform the sketch into place ---------------- #
    print("4 transform")
    ys, xs = np.nonzero(sketch > 0.05)
    quad0 = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()],
                      [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float32)
    Minv = np.linalg.inv(np.vstack([M_wrong, [0, 0, 1]]))
    n_tr = nfr["transform"]
    for f in range(n_tr):
        t = np.clip((f / max(1, n_tr - 1) - 0.2) / 0.7, 0, 1)  # pause after selecting, then ease
        t = t * t * (3 - 2 * t)
        M = (np.eye(3) * (1 - t) + Minv * t)[:2]  # matrix lerp is fine for small transforms
        layer = cv2.warpAffine(sketch, M, (w, h), flags=cv2.INTER_LINEAR)
        frame = sketch_frame(cons_op=0.5 * (1 - t), sk_layer=layer)
        emit(dashed_rect(frame, apply_affine(quad0, M), f / 3))
    sketch = cv2.warpAffine(sketch, Minv[:2], (w, h), flags=cv2.INTER_LINEAR)
    cons[:] = 0
    emit(sketch_frame(), still="transform")

    # ---------------- 5. inking ---------------- #
    print("5 inking")
    reveal = np.zeros((h, w), np.uint8)
    ink_r = 2.2 * sc + 2
    ink_strokes = [Stroke(p) for p in order_nn(split(skel, 60 * sc, 220 * sc, rng))]
    n_ink = nfr["ink"]
    ink_state = {"f": 0}

    def ink_tick(pt, k):
        t = ink_state["f"] / max(1, n_ink - 1)
        ink_state["f"] += 1
        sk_op = 1.0 - 0.7 * min(1, t / 0.06)               # lower sketch opacity
        sk_op *= 1.0 - np.clip((t - 0.92) / 0.08, 0, 1)    # hide it at the end
        base = blend(paper, sk_full, sketch * sk_op)
        emit(blend(base, line_col, line_alpha * (reveal > 0)), pt)

    play(ink_strokes, n_ink, lambda s, a, b: stamp_line(reveal, s.pts[a:b + 1], 2 * ink_r), ink_tick)
    emit(with_lines(paper), still="lineart")

    # ---------------- 6. base colours ---------------- #
    # Each area is filled one piece at a time, the colour spreading outward from
    # where the brush starts (geodesic distance inside the shape), like painting
    # a region in rather than scattering strokes.
    print("6 base colours")
    base_layer = paper.copy()
    comps = []
    for c in sorted(range(len(base_cols)), key=lambda c: -(mat_idx == c).sum()):
        n, lab, stats, cents = cv2.connectedComponentsWithStats((mat_idx == c).astype(np.uint8), connectivity=4)
        parts = [(j, stats[j], cents[j]) for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= 25]
        # Visit this colour's pieces in a natural order: nearest next piece.
        here = np.array([0.0, 0.0]) if not comps else comps[-1]["cent"]
        while parts:
            k = int(np.argmin([np.sum((p[2] - here) ** 2) for p in parts]))
            j, st, ce = parts.pop(k)
            comps.append(dict(c=c, lab=lab, j=j, st=st, cent=ce))
            here = ce
    weights_c = np.array([np.sqrt(cp["st"][cv2.CC_STAT_AREA]) for cp in comps])
    n_blk = nfr["block"]
    alloc = np.maximum(0, np.floor(weights_c / weights_c.sum() * n_blk)).astype(int)
    cursor_pt = None
    fill_noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6 * sc)
    fill_noise = (fill_noise - fill_noise.mean()) / (fill_noise.std() + 1e-6)
    written = 0
    for cp, nf in zip(comps, alloc):
        x, y, cw, ch = (int(v) for v in cp["st"][:4])
        mask = cp["lab"][y:y + ch, x:x + cw] == cp["j"]
        col = wrong_cols[cp["c"]]
        if nf <= 0:  # tiny piece: just dab it in
            base_layer[y:y + ch, x:x + cw][mask] = col
            continue
        yy_, xx_ = np.nonzero(mask)
        ref = np.array(cursor_pt if cursor_pt is not None else (x, y), np.float32)
        k = int(np.argmin((xx_ + x - ref[0]) ** 2 + (yy_ + y - ref[1]) ** 2))
        seed = (int(yy_[k]), int(xx_[k]))
        dist, _ = MCP_Geometric(np.where(mask, 1.0, np.inf)).find_costs([seed])
        dist = np.where(mask, dist, np.inf)
        dmax = float(dist[mask].max()) + 1e-6
        nz = fill_noise[y:y + ch, x:x + cw] * min(dmax * 0.06, 14 * sc)
        for f in range(nf):
            t = (f + 1) / nf
            t = 1 - (1 - t) ** 1.6  # quick start, careful finish near the edges
            thr = dmax * t
            cur = mask & (dist + nz <= thr) if f < nf - 1 else mask
            base_layer[y:y + ch, x:x + cw][cur] = col
            front = np.nonzero(mask & (np.abs(dist - thr) < max(2.0, dmax / nf)))
            if len(front[0]):
                i = int(rng.integers(len(front[0])))
                cursor_pt = (front[1][i] + x, front[0][i] + y)
            emit(with_lines(base_layer), cursor_pt, 6 * sc + 3)
            written += 1
    for _ in range(max(0, n_blk - written)):
        emit(with_lines(base_layer))
    emit(with_lines(flats_wrong), still="base-colours")

    # ---------------- 7. colour adjust ---------------- #
    print("7 colour adjust")
    n_col = nfr["colour"]
    knobs = (rng.uniform(-18, 18), rng.uniform(12, 25), rng.uniform(-8, 8))
    for f in range(n_col):
        t = np.clip((f / max(1, n_col - 1) - 0.15) / 0.65, 0, 1)
        t = t * t * (3 - 2 * t)
        cur = blend(flats_wrong, flats, np.full((h, w), t, np.float32))
        emit(hsl_panel(with_lines(cur), *(k * t for k in knobs)))
    flats_frame = with_lines(flats)
    emit(flats_frame, still="flats")

    # ---------------- 8/9. shading and rendering ---------------- #
    shade_target = with_lines(np.minimum(img, flats))  # multiply-style shadows only

    def paint_towards(src, dst, n, still):
        diff = np.abs(dst.astype(np.int16) - src.astype(np.int16)).sum(2).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (0, 0), 4) + 0.5
        weights = (diff / diff.sum()).ravel()
        acc = np.zeros((h, w), np.float32)
        dab_r = max(6, int(min(w, h) / 26))
        per = max(4, int(3.0 * (w * h) / (np.pi * dab_r ** 2) / max(1, n)))
        focus = None
        for f in range(n):
            if f % max(1, n // 14) == 0:  # work on one area at a time
                focus = np.unravel_index(rng.choice(len(weights), p=weights), (h, w))
            local = np.zeros((h, w), np.float32)
            last = None
            for _ in range(per):
                if rng.random() < 0.75:
                    y = int(np.clip(rng.normal(focus[0], dab_r * 4), 0, h - 1))
                    x = int(np.clip(rng.normal(focus[1], dab_r * 4), 0, w - 1))
                else:
                    y, x = np.unravel_index(rng.choice(len(weights), p=weights), (h, w))
                cv2.circle(local, (int(x), int(y)), int(dab_r * rng.uniform(0.5, 1.2)), 0.4, -1, cv2.LINE_AA)
                last = (x, y)
            acc = np.clip(acc + cv2.GaussianBlur(local, (0, 0), dab_r / 3), 0, 1)
            t = (f + 1) / n
            emit(blend(src, dst, np.maximum(acc, t ** 3)), last, dab_r)
        emit(dst, still=still)

    print("8 shading")
    paint_towards(flats_frame, shade_target, nfr["shade"], "shading")
    print("9 rendering")
    paint_towards(shade_target, img, nfr["render"], "render")

    # ---------------- 10. background + hold ---------------- #
    final = img
    if args.bg_final:
        print("10 background")
        bg = load_image(args.bg_final, size=(w, h))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        diag = (xx / w + yy / h) / 2
        n_bg = max(1, nfr["bg"])
        for f in range(n_bg):
            t = (f + 1) / n_bg
            emit(blend(img, bg, np.clip((t * 1.25 - diag) / 0.25, 0, 1)))
        final = bg
    for _ in range(nfr["hold"]):
        emit(final)
    emit(final, still="final")
    out.close()
    print(f"Done: {args.output} ({out.count / args.fps:.1f}s, {out.count} frames)")


if __name__ == "__main__":
    main()
