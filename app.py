import cv2
import numpy as np
from typing import List, Tuple
import os
import time

# ─────────────────────────────
# CONSTANTS & MODES
# ─────────────────────────────
WINDOW_NAME = 'Webcam Viewer (Fullscreen)'

# Top-level modes
MODE_VIEW       = 0   # raw camera only
MODE_GEOMETRIC  = 1   # transforms with trackbars
MODE_COLOR      = 2   # color space conversions
MODE_TONE       = 3   # brightness & contrast
MODE_HIST       = 4   # histogram visualization
MODE_BLUR       = 5   # gaussian blur
MODE_BILATERAL  = 6   # bilateral filter
MODE_CANNY      = 7   # canny edge detection
MODE_HOUGH      = 8   # Hough line detection
MODE_PANO       = 9   # basic panorama (left -> right)
MODE_AR         = 10  # Augmented Reality (ArUco + OBJ)  ← AR from second script
MODE_CALIB      = 11  # Chessboard capture & calibration  ← calibration from second script
MODE_CALIB_RES  = 12  # Show calibration summary/result
MODE_UNDISTORT  = 13  # Live undistort preview

# Sub-modes inside GEOMETRIC
SUB_RIGID       = 0  # rotation + translation
SUB_SIMILARITY  = 1  # rotation + translation + uniform scale

# Sub-modes inside COLOR
COLOR_RGB  = 0  # (actually BGR from OpenCV camera, displayed as-is)
COLOR_GRAY = 1
COLOR_HSV  = 2

# Trackbar configurations for GEOMETRIC: (label, min_val, max_val, default_value)
GEOM_TRACKBARS = {
    'Angle':       ('Angle', -180, 180, 0),             # degrees
    'Translate X': ('Translate X', -300, 300, 0),       # pixels
    'Translate Y': ('Translate Y', -200, 200, 0),       # pixels
    'Scale':       ('Scale', 20, 300, 100),             # 0.20x..3.00x (100=>1.00x)
    'Spin Speed':  ('Spin Speed (deg/s)', 0, 360, 90),  # auto-rotation
}

# Trackbar configurations for TONE (Brightness/Contrast)
TONE_TRACKBARS = {
    'Brightness': ('Brightness', -200, 200, 0),
    'Contrast':   ('Contrast (x0.01)', 20, 400, 100),   # 0.20x .. 4.00x (100=>1.00x)
}
# --- helpers (once, top of file) ---
def R_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def R_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)


def R_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)




# Trackbars for GAUSSIAN BLUR
BLUR_TRACKBARS = {
    'Kernel': ('Kernel (odd px)', 1, 99, 5),    # coerced to odd
    'SigmaX': ('SigmaX (x0.1)',   0, 300, 10),  # 0 => auto; 10 => 1.0
}

# Trackbars for BILATERAL FILTER
BILAT_TRACKBARS = {
    'Diameter':   ('Diameter (px; 0=auto)', 0, 25, 9),  # neighborhood diameter
    'SigmaColor': ('SigmaColor',            1, 250, 75),
    'SigmaSpace': ('SigmaSpace',            1, 250, 75),
}

# Trackbars for CANNY EDGE
CANNY_TRACKBARS = {
    'Lower':        ('Lower Thresh', 0, 500, 50),
    'Upper':        ('Upper Thresh', 0, 500, 150),
    'ApertureIdx':  ('Aperture (0:3|1:5|2:7)', 0, 2, 0),
    'L2Grad':       ('L2Gradient (0/1)', 0, 1, 0),
    'PreBlur':      ('PreBlur k (odd)', 1, 15, 1),
}

# Trackbars for HOUGH LINES (uses its own Canny first)
HOUGH_TRACKBARS = {
    # Canny pre-processing
    'HLower':      ('Canny Lower', 0, 500, 80),
    'HUpper':      ('Canny Upper', 0, 500, 200),
    'HApertureId': ('Aperture (0:3|1:5|2:7)', 0, 2, 0),
    'HL2Grad':     ('L2Gradient (0/1)', 0, 1, 0),
    'HPreBlur':    ('PreBlur k (odd)', 1, 15, 1),

    # Hough parameters
    'UseP':        ('Use Probabilistic (0/1)', 0, 1, 1),  # 1 => HoughLinesP, 0 => HoughLines
    'Rho':         ('Rho (px)', 1, 5, 1),
    'ThetaDeg':    ('Theta (deg per step)', 1, 5, 1),     # step granularity
    'Thresh':      ('Votes Threshold', 1, 300, 80),

    # Only for Probabilistic
    'MinLen':      ('Min Line Length', 0, 500, 50),
    'MaxGap':      ('Max Line Gap', 0, 200, 10),
}

# Trackbars for PANORAMA (basic L→R strip stitch)
PANO_TRACKBARS = {
    'Overlap': ('Overlap (px)', 10, 300, 80),   # blend width across seams
}

# ─────────────────────────────
# UI HELPERS (FULLSCREEN)
# ─────────────────────────────
def _apply_fullscreen():
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

def _exit_fullscreen():
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except cv2.error:
        pass

def create_window(fullscreen: bool = True):
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    except cv2.error:
        pass
    if fullscreen:
        _apply_fullscreen()

def destroy_window():
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass

def _setup_trackbars(config: dict, fullscreen: bool = True):
    destroy_window()
    create_window(fullscreen=fullscreen)
    def _noop(x): pass
    for _, (label, min_val, max_val, default_value) in config.items():
        range_max   = (max_val - min_val) if min_val != 0 else max_val
        default_pos = (default_value - min_val) if min_val != 0 else default_value
        default_pos = max(0, min(range_max, default_pos))
        cv2.createTrackbar(label, WINDOW_NAME, default_pos, range_max, _noop)

def setup_geometric_ui(fullscreen: bool = True):
    _setup_trackbars(GEOM_TRACKBARS, fullscreen=fullscreen)

def setup_tone_ui(fullscreen: bool = True):
    _setup_trackbars(TONE_TRACKBARS, fullscreen=fullscreen)

def setup_blur_ui(fullscreen: bool = True):
    _setup_trackbars(BLUR_TRACKBARS, fullscreen=fullscreen)

def setup_bilateral_ui(fullscreen: bool = True):
    _setup_trackbars(BILAT_TRACKBARS, fullscreen=fullscreen)

def setup_canny_ui(fullscreen: bool = True):
    _setup_trackbars(CANNY_TRACKBARS, fullscreen=fullscreen)

def setup_hough_ui(fullscreen: bool = True):
    _setup_trackbars(HOUGH_TRACKBARS, fullscreen=fullscreen)

def setup_pano_ui(fullscreen: bool = True):
    _setup_trackbars(PANO_TRACKBARS, fullscreen=fullscreen)

def get_trackbar_value(name: str, config: dict) -> int:
    label, min_val, _, default_val = config[name]
    try:
        pos = cv2.getTrackbarPos(label, WINDOW_NAME)
    except cv2.error:
        return default_val
    return pos + min_val if min_val != 0 else pos

def set_trackbar_value(name: str, real_value: int, config: dict):
    label, min_val, max_val, _ = config[name]
    real_value = max(min_val, min(max_val, real_value))
    pos = real_value - min_val if min_val != 0 else real_value
    range_max = (max_val - min_val) if min_val != 0 else max_val
    pos = max(0, min(range_max, pos))
    try:
        cv2.setTrackbarPos(label, WINDOW_NAME, pos)
    except cv2.error:
        pass

def wrap_angle_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0

def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else k + 1

# ─────────────────────────────
# TEXT / MENU (from second script style)
# ─────────────────────────────
def _draw_text(frame, text, pos=(20, 50), scale=0.8, color=(255,255,255)):
    th = max(1, int(scale * 2))
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), th+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

def draw_bottom_menu(
    frame: np.ndarray,
    lines: List[str],
    pad_x: int = 18,
    pad_y: int = 14,
    line_gap: int = 8,
    font_scale: float = 0.72,
    thickness: int = 2,
    fg_color=(255, 255, 255),
    origin=None,
) -> None:
    h, w = frame.shape[:2]
    max_w = w - 2 * pad_x
    font = cv2.FONT_HERSHEY_SIMPLEX
    leading = 1.25

    def wrap_text(text: str) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        out, cur = [], words[0]
        for word in words[1:]:
            test = cur + " " + word
            (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_w:
                cur = test
            else:
                out.append(cur)
                cur = word
        out.append(cur)
        return out

    wrapped: List[str] = []
    for ln in lines:
        wrapped.extend(wrap_text(ln))

    (sample_w, sample_h), sample_base = cv2.getTextSize("Ag", font, font_scale, thickness)
    base_h = sample_h
    line_height = int(base_h * leading) + line_gap

    block_height = pad_y * 2 + line_height * len(wrapped)
    max_block_h = int(h * 0.35)
    if block_height > max_block_h:
        shrink = max(0.5, max_block_h / float(block_height))
        font_scale *= shrink
        (sample_w, sample_h), sample_base = cv2.getTextSize("Ag", font, font_scale, thickness)
        base_h = sample_h
        line_height = int(base_h * leading) + line_gap
        block_height = pad_y * 2 + line_height * len(wrapped)

    y0 = h - block_height
    x0 = 0
    x1 = w
    y1 = h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    y = y0 + pad_y + base_h
    for text in wrapped:
        cv2.putText(frame, text, (pad_x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (pad_x, y), font, font_scale, fg_color, thickness, cv2.LINE_AA)
        y += line_height

# ─────────────────────────────
# COLOR / TONE / HISTOGRAM
# ─────────────────────────────
def convert_color(frame_bgr: np.ndarray, color_mode: int) -> np.ndarray:
    if color_mode == COLOR_RGB:
        return frame_bgr
    elif color_mode == COLOR_GRAY:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif color_mode == COLOR_HSV:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    else:
        return frame_bgr

def apply_brightness_contrast(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.addWeighted(frame, alpha, np.zeros_like(frame), 0, beta)

def make_hist_canvas(frame_bgr: np.ndarray, mode: str = "color", log_scale: bool = False) -> np.ndarray:
    W, H = 640, 360
    margin = 24
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (margin, margin), (W-margin, H-margin), (60,60,60), 1)

    bins = 256
    x_axis = np.linspace(margin, W - margin, bins).astype(np.int32)

    if mode == "gray":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
        if log_scale:
            hist = np.log1p(hist)
        hist = hist / (hist.max() + 1e-6)
        y = (H - margin) - (H - 2*margin) * hist
        pts = np.vstack([x_axis, y]).T.astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(canvas, tuple(pts[i-1]), tuple(pts[i]), (220, 220, 220), 2, cv2.LINE_AA)
        label = "GRAY"
    else:
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        for ch, col in enumerate(colors):
            hist = cv2.calcHist([frame_bgr], [ch], None, [bins], [0, 256]).flatten()
            if log_scale:
                hist = np.log1p(hist)
            hist = hist / (hist.max() + 1e-6)
            y = (H - margin) - (H - 2*margin) * hist
            pts = np.vstack([x_axis, y]).T.astype(np.int32)
            for i in range(1, len(pts)):
                cv2.line(canvas, tuple(pts[i-1]), tuple(pts[i]), col, 1, cv2.LINE_AA)
        label = "BGR"

    cv2.putText(canvas, f"Histogram [{label}] {'(log)' if log_scale else ''}",
                (margin, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "0",   (margin-5, H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "255", (W-margin-38, H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1, cv2.LINE_AA)
    return canvas

# ─────────────────────────────
# RESETS
# ─────────────────────────────
def reset_geometric_ui():
    set_trackbar_value('Angle',        0,   GEOM_TRACKBARS)
    set_trackbar_value('Translate X',  0,   GEOM_TRACKBARS)
    set_trackbar_value('Translate Y',  0,   GEOM_TRACKBARS)
    set_trackbar_value('Scale',        100, GEOM_TRACKBARS)  # 1.00x
    set_trackbar_value('Spin Speed',   0,   GEOM_TRACKBARS)

def reset_canny_ui():
    set_trackbar_value('Lower',       50,  CANNY_TRACKBARS)
    set_trackbar_value('Upper',       150, CANNY_TRACKBARS)
    set_trackbar_value('ApertureIdx', 0,   CANNY_TRACKBARS)  # -> 3
    set_trackbar_value('L2Grad',      0,   CANNY_TRACKBARS)
    set_trackbar_value('PreBlur',     1,   CANNY_TRACKBARS)

def reset_hough_ui():
    set_trackbar_value('HLower',      80,  HOUGH_TRACKBARS)
    set_trackbar_value('HUpper',      200, HOUGH_TRACKBARS)
    set_trackbar_value('HApertureId', 0,   HOUGH_TRACKBARS)
    set_trackbar_value('HL2Grad',     0,   HOUGH_TRACKBARS)
    set_trackbar_value('HPreBlur',    1,   HOUGH_TRACKBARS)
    set_trackbar_value('UseP',        1,   HOUGH_TRACKBARS)
    set_trackbar_value('Rho',         1,   HOUGH_TRACKBARS)
    set_trackbar_value('ThetaDeg',    1,   HOUGH_TRACKBARS)
    set_trackbar_value('Thresh',      80,  HOUGH_TRACKBARS)
    set_trackbar_value('MinLen',      50,  HOUGH_TRACKBARS)
    set_trackbar_value('MaxGap',      10,  HOUGH_TRACKBARS)

# ─────────────────────────────
# HOUGH HELPERS
# ─────────────────────────────
def _draw_standard_hough_lines(bgr, lines):
    if lines is None:
        return 0
    count = 0
    for l in lines:
        rho, theta = l[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv2.line(bgr, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        count += 1
    return count

# ─────────────────────────────
# PANORAMA (LEFT → RIGHT) - same as your first script
# ─────────────────────────────
class LeftRightPanorama:
    def __init__(self, overlap_px: int = 80):
        self.canvas: np.ndarray = None
        self.prev: np.ndarray = None
        self.cur_x: int = 0
        self.overlap_px: int = int(max(10, overlap_px))
        self.last_status: str = ""

    def set_overlap(self, overlap_px: int):
        self.overlap_px = int(max(10, overlap_px))

    def reset(self):
        self.canvas = None
        self.prev = None
        self.cur_x = 0
        self.last_status = "Reset."

    def _match_orb_ratio(self, a: np.ndarray, b: np.ndarray, max_features=4000, ratio=0.75):
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7,
                             edgeThreshold=15, scaleFactor=1.2, nlevels=8)
        kpsA, desA = orb.detectAndCompute(grayA, None)
        kpsB, desB = orb.detectAndCompute(grayB, None)
        if desA is None or desB is None or len(kpsA) < 8 or len(kpsB) < 8:
            return np.empty((0,2)), np.empty((0,2))
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(desA, desB, k=2)
        good = []
        for p in knn:
            if len(p) < 2: continue
            m, n = p
            if m.distance < ratio * n.distance:
                good.append(m)
        if not good: return np.empty((0,2)), np.empty((0,2))
        good = sorted(good, key=lambda m: m.distance)
        keep = max(30, int(0.25 * len(good)))
        good = good[:keep]
        ptsA = np.float64([kpsA[m.queryIdx].pt for m in good])  # new
        ptsB = np.float64([kpsB[m.trainIdx].pt for m in good])  # prev
        return ptsA, ptsB

    def _blend_paste(self, img: np.ndarray, paste_x: int, blend_w: int):
        h, w = img.shape[:2]
        need_w = max(self.canvas.shape[1], paste_x + w)
        if need_w > self.canvas.shape[1]:
            grown = np.zeros((h, need_w, 3), dtype=np.uint8)
            grown[:, :self.canvas.shape[1]] = self.canvas
            self.canvas = grown
        if blend_w > 0:
            x0 = paste_x
            x1 = paste_x + blend_w
            left  = self.canvas[:, x0:x1, :].astype(np.float32)
            right = img[:, :blend_w, :].astype(np.float32)
            alpha = np.linspace(0.0, 1.0, blend_w, dtype=np.float32)[None, :, None]
            mix = left * (1.0 - alpha) + right * alpha
            self.canvas[:, x0:x1, :] = np.clip(mix, 0, 255).astype(np.uint8)
        self.canvas[:, paste_x + blend_w: paste_x + w, :] = img[:, blend_w:, :]

    def add_image(self, img_bgr: np.ndarray):
        if self.canvas is None:
            self.canvas = img_bgr.copy()
            self.prev = img_bgr.copy()
            self.cur_x = img_bgr.shape[1]
            self.last_status = "Anchor captured."
            return
        ptsNew, ptsPrev = self._match_orb_ratio(img_bgr, self.prev)
        if len(ptsNew) < 12:
            self.last_status = f"Not enough matches ({len(ptsNew)})."
            return
        dx_raw = np.median(ptsNew[:, 0] - ptsPrev[:, 0])
        dy_raw = np.median(ptsNew[:, 1] - ptsPrev[:, 1])
        if dx_raw >= -3.0:
            self.last_status = f"Ignored: not moving right (Δx={dx_raw:.1f})."
            return
        s = float(np.median(ptsPrev[:, 0] - ptsNew[:, 0]))
        h, w = img_bgr.shape[:2]
        overlap_est = int(round(np.clip(w - s, 10, w - 10)))
        blend_w = int(min(self.overlap_px, overlap_est))
        ty = int(round(np.clip(dy_raw * (-1.0), -3, 3)))
        if ty != 0:
            M = np.float32([[1, 0, 0], [0, 1, ty]])
            img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        paste_x = self.cur_x - overlap_est
        self._blend_paste(img_bgr, paste_x, blend_w)
        self.cur_x = paste_x + w
        self.prev = img_bgr.copy()
        self.last_status = f"Added: matches={len(ptsNew)}, overlap≈{overlap_est}px, blend={blend_w}px"

# ─────────────────────────────
# AR & CALIBRATION (from second script)
# ─────────────────────────────
def _load_calibration():
    """Loads intrinsic matrix and distortion coefficients from calibration.npz."""
    if os.path.exists('calibration.npz'):
        try:
            data = np.load('calibration.npz')
            mtx, dist = data['mtx'], data['dist']
            print("Calibration data loaded from 'calibration.npz'.")
            return mtx, dist
        except Exception as e:
            print(f"ERROR: Could not load calibration data: {e}")
    print("WARNING: 'calibration.npz' not found. AR/undistort need calibration.")
    return None, None

def _init_aruco():
    """Create a detector compatible with both new and legacy ArUco APIs."""
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return ('new', detector, aruco_dict, params)
    except AttributeError:
        print("Warning: Legacy ArUco API detected.")
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters_create()
        return ('old', None, aruco_dict, params)

def _load_obj_simple(filename: str, scale_factor=0.0005):
    """Minimal OBJ loader (from second script): center and scale."""
    if not os.path.exists(filename):
        print(f"ERROR: Model file '{filename}' not found.")
        return np.empty((0,3), np.float32), []
    vertices, faces = [], []
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()[1:4]
                    if len(parts) == 3:
                        try:
                            x, y, z = map(float, parts)
                            vertices.append([x, y, z])
                        except ValueError:
                            continue
                elif line.startswith('f '):
                    idxs = []
                    for part in line.strip().split()[1:]:
                        idx_str = part.split('/')[0]
                        try:
                            idxs.append(int(idx_str) - 1)
                        except (ValueError, IndexError):
                            continue
                    if len(idxs) >= 3:
                        # triangulate fan
                        for t in range(1, len(idxs)-1):
                            faces.append([idxs[0], idxs[t], idxs[t+1]])
        if not vertices:
            print("ERROR: No vertices found in the OBJ.")
            return np.empty((0,3), np.float32), []
        verts_np = np.array(vertices, dtype=np.float32)
        center = np.mean(verts_np, axis=0)
        verts_np = (verts_np - center) * scale_factor
        print(f"Loaded OBJ: {len(vertices)} verts, {len(faces)} tris (centered & scaled).")
        return verts_np.astype(np.float32), faces
    except Exception as e:
        print(f"Error loading OBJ file {filename}: {e}")
        return np.empty((0,3), np.float32), []

def _draw_ar_wireframe(frame, detector_kind, detector, aruco_dict, params,
                       model_vertices: np.ndarray, model_faces: list,
                       cam_mtx, dist, marker_size_m=0.05):
    out = frame.copy()
    if cam_mtx is None or dist is None:
        return out

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if detector_kind == 'new':
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return out

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_m, cam_mtx, dist)
    cv2.aruco.drawDetectedMarkers(out, corners, ids)

    if model_vertices.size == 0 or len(model_faces) == 0:
        return out

    # Rotation correction matrix
    R_correction, _ = cv2.Rodrigues(np.array([[0, np.pi/2, 0]], dtype=np.float32))  
    # -90° around X → put feet on marker
    # adjust with np.pi/2 on Y or Z if you want sideways rotation
    # Step 1: put head on marker
    Rx = R_x(np.pi/2)

    # Step 2: yaw so head points toward camera
    Ry = R_y(np.pi*2)   # try -np.pi/2 if facing sideways
    Rz = R_z(-np.pi/2)    # roll correction if needed

    R_align = Ry @ Rx
    # R_align = R_y(-np.pi/2) 
    for i in range(len(ids)):
        # R_marker, _ = cv2.Rodrigues(rvecs[i][0])
        # R_final = R_marker @ R_correction
        R_marker, _ = cv2.Rodrigues(rvecs[i][0])
        R_final = R_marker @ R_align
        rvec_final, _ = cv2.Rodrigues(R_final)
        scale_factor = 1.5   # adjust 1.0 = normal, >1 = bigger, <1 = smaller
        center = np.mean(model_vertices, axis=0)
        scaled_vertices = (model_vertices - center) * scale_factor + center
        pts, _ = cv2.projectPoints(scaled_vertices, rvec_final, tvecs[i][0], cam_mtx, dist)
        pts = pts.reshape(-1, 2)
        for tri in model_faces:
            poly = np.int32(pts[tri]).reshape(-1, 2)
            cv2.fillConvexPoly(out, poly, (0, 255, 0))   # solid fill
            cv2.polylines(out, [poly], True, (0, 0, 0), 1, cv2.LINE_AA)  # black outline


    return out

# ─────────────────────────────
# CALIBRATION PIPELINE (from second script)
# ─────────────────────────────
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM  = 25
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OBJP = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
OBJP[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

def _run_calibration_step(frame, objpoints, imgpoints, last_capture_time, target_images=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=cv2.CALIB_CB_FAST_CHECK)
    disp = frame.copy()
    if ret:
        cv2.drawChessboardCorners(disp, CHESSBOARD_SIZE, corners, ret)
        if (time.time() - last_capture_time[0]) > 2 and len(objpoints) < target_images:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            objpoints.append(OBJP.copy())
            imgpoints.append(refined)
            last_capture_time[0] = time.time()
            print(f"Captured calibration image {len(objpoints)}/{target_images}")
    _draw_text(disp, f"Calibrating... {len(objpoints)}/{target_images}", (50, 50), 0.9, (0, 200, 255))
    return disp

def _finalize_calibration(objpoints, imgpoints, imshape):
    gray_h, gray_w = imshape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (gray_w, gray_h), None, None)
    mean_error = 0.0
    for i in range(len(objpoints)):
        proj_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], proj_pts, cv2.NORM_L2) / len(proj_pts)
        mean_error += error
    final_error = mean_error / max(1, len(objpoints))
    np.savez('calibration.npz', mtx=mtx, dist=dist, error=final_error)
    print(f"Calibration complete. Reprojection error: {final_error:.4f}")
    return {'mtx': mtx, 'dist': dist, 'error': final_error}

def _show_calibration_summary(frame, calib_res):
    disp = np.zeros_like(frame)
    y = 60
    _draw_text(disp, "CALIBRATION COMPLETE", (50, y), 1.0, (0, 255, 0)); y += 60
    _draw_text(disp, f"Reprojection Error: {calib_res.get('error', 0):.4f}", (60, y), 0.8); y += 50
    _draw_text(disp, "Intrinsic Matrix:", (60, y), 0.7); y += 30
    if 'mtx' in calib_res:
        for row in calib_res['mtx']:
            _draw_text(disp, f"[{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]", (80, y), 0.65); y += 30
    y += 10
    _draw_text(disp, "Press 'U' for undistorted view.", (60, y), 0.8, (255, 255, 0))
    return disp

def _apply_undistort(frame, mtx, dist):
    if mtx is None:
        out = frame.copy()
        _draw_text(out, "Calibrate first with [K]", (50, 50), 0.9, (0, 0, 255))
        return out
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
    x, y, ww, hh = roi
    if ww > 0 and hh > 0:
        undistorted = undistorted[y:y + hh, x:x + ww]
    return cv2.resize(undistorted, (w, h))

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    # Load model (from second script style)
    MODEL_PATH = "assets/trex_model.obj"
    model_vertices, model_faces = _load_obj_simple(MODEL_PATH, scale_factor=0.0005)

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # AR init (from second script)
    cam_mtx, dist = _load_calibration()
    aruco_kind, aruco_detector, aruco_dict, aruco_params = _init_aruco()

    # Calibration state
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    last_capture_time = [time.time()]
    TARGET_IMAGES = 20
    calib_results = {}

    fullscreen = True
    create_window(fullscreen=fullscreen)

    mode = MODE_VIEW
    submode = SUB_RIGID
    color_mode = COLOR_RGB
    auto_spin = False

    hist_gray_only = False
    hist_log = False

    pano = LeftRightPanorama()

    prev_ticks = cv2.getTickCount()
    tick_freq  = cv2.getTickFrequency()

    print(
        "\nControls:"
        "\n  f         : toggle fullscreen"
        "\n  v         : quick View mode"
        "\n  g         : enter/exit Geometric"
        "\n  c         : enter/exit Color"
        "\n  b         : enter/exit Brightness/Contrast"
        "\n  h         : enter/exit Histogram"
        "\n  k         : enter/exit Calibration (chessboard)"
        "\n  u         : enter/exit Undistort live view"
        "\n  i         : enter/exit Bilateral Filter"
        "\n  e         : enter/exit Canny Edge Detection"
        "\n  t         : enter/exit Hough Line Detection"
        "\n  y         : enter/exit Panorama (basic L→R)"
        "\n  a         : enter/exit AR (OBJ wireframe)"
        "\n  ESC       : quit\n"
        "\nIn Geometric mode:"
        "\n  m         : toggle Rigid <-> Similarity"
        "\n  SPACE     : toggle auto-rotation"
        "\n  a / d     : rotate -5° / +5°"
        "\n  0         : RESET (angle/tx/ty/scale; spin OFF)\n"
        "In Color mode: 1: RGB   2: Gray   3: HSV\n"
        "Histogram: 1: BGR  2: GRAY  3/L: toggle log\n"
        "Bilateral reset: '0'   Gaussian reset: '0'\n"
        "Canny reset: '0'   Hough reset: '0'\n"
        "Panorama: n=add  r=reset  s=save\n"
        "Calibration: capture auto every 2s; collects 20 images; auto-saves calibration.npz.\n"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rows, cols = frame.shape[:2]
        display = frame.copy()

        cur_ticks = cv2.getTickCount()
        dt = (cur_ticks - prev_ticks) / tick_freq
        prev_ticks = cur_ticks

        if mode == MODE_VIEW:
            draw_bottom_menu(display, [
                "Mode: View (raw camera)  |  'f' fullscreen",
                "Press [G]eometric, [C]olor, [B]right/Contrast, [H]istogram, [I] Bilateral, [E] Canny, [T] Hough, [Y] Panorama, [A]R, [K] Calibrate, [U] Undistort",
                "ESC to quit",
            ])

        elif mode == MODE_GEOMETRIC:
            angle      = float(get_trackbar_value('Angle', GEOM_TRACKBARS))
            tx         = get_trackbar_value('Translate X', GEOM_TRACKBARS)
            ty         = get_trackbar_value('Translate Y', GEOM_TRACKBARS)
            spin_speed = float(get_trackbar_value('Spin Speed', GEOM_TRACKBARS))

            if auto_spin and spin_speed > 0:
                angle = wrap_angle_deg(angle + spin_speed * dt)
                set_trackbar_value('Angle', int(round(angle)), GEOM_TRACKBARS)

            if submode == SUB_RIGID:
                scale = 1.0
                sub_text = "Rigid"
            else:
                scale = get_trackbar_value('Scale', GEOM_TRACKBARS) / 100.0
                sub_text = f"Similarity (Scale {scale:.2f}x)"

            center = (cols/2.0, rows/2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            display = cv2.warpAffine(
                frame, M, (cols, rows),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(47, 53, 66)
            )

            spin_text = "ON" if auto_spin and spin_speed > 0 else "OFF"
            draw_bottom_menu(display, [
                f"Mode: Geometric · {sub_text}  |  'f' fullscreen",
                "[V]iew   [C]olor   [B]right/Contrast   [H]istogram   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   m: Rigid/Similarity   ESC: quit",
                "SPACE: auto-rotate   a/d: -5°/+5°   0: RESET (angle/tx/ty/scale; spin OFF)",
                f"Angle:{int(round(angle))}°  Tx:{tx}  Ty:{ty}  Scale:{scale:.2f}x  Spin:{spin_text} ({int(spin_speed)}°/s)"
            ])

        elif mode == MODE_COLOR:
            converted = convert_color(frame, color_mode)
            display = converted
            color_text = (
                "RGB (original BGR)" if color_mode == COLOR_RGB else
                "Gray" if color_mode == COLOR_GRAY else
                "HSV (raw HSV visualization)"
            )
            draw_bottom_menu(display, [
                f"Mode: Color · {color_text}  |  'f' fullscreen",
                "[G]eometric   [V]iew   [B]right/Contrast   [H]istogram   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   ESC: quit",
                "1: RGB   2: Gray   3: HSV"
            ])

        elif mode == MODE_TONE:
            beta  = get_trackbar_value('Brightness', TONE_TRACKBARS)
            alpha = get_trackbar_value('Contrast',   TONE_TRACKBARS) / 100.0
            display = apply_brightness_contrast(frame, alpha, beta)
            draw_bottom_menu(display, [
                f"Mode: Brightness/Contrast  (alpha={alpha:.2f}x, beta={beta})  |  'f' fullscreen",
                "[G]eometric   [C]olor   [V]iew   [H]istogram   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   ESC: quit",
                "Trackbars: Brightness [-200..200], Contrast [0.20..4.00]",
                "Press '0' to reset (Brightness=0, Contrast=1.00x)"
            ])

        elif mode == MODE_BLUR:
            k  = get_trackbar_value('Kernel', BLUR_TRACKBARS)
            sx = get_trackbar_value('SigmaX', BLUR_TRACKBARS) / 10.0
            k = max(1, _ensure_odd(k))
            sigma_x = float(sx)
            display = cv2.GaussianBlur(frame, (k, k), sigmaX=sigma_x, sigmaY=0)
            draw_bottom_menu(display, [
                f"Mode: Gaussian Blur  |  Kernel={k}  SigmaX={sigma_x:.1f}  |  'f' fullscreen",
                "[G]eometric   [C]olor   [B]right/Contrast   [H]istogram   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   [V]iew   ESC: quit",
                "Press '0' to reset (Kernel=5, SigmaX=1.0)   ·   'k' to exit"
            ])

        elif mode == MODE_BILATERAL:
            d   = get_trackbar_value('Diameter',   BILAT_TRACKBARS)
            sc  = get_trackbar_value('SigmaColor', BILAT_TRACKBARS)
            ss  = get_trackbar_value('SigmaSpace', BILAT_TRACKBARS)
            display = cv2.bilateralFilter(frame, d if d > 0 else 0, sc, ss)
            draw_bottom_menu(display, [
                f"Mode: Bilateral Filter  |  d={d}  SigmaColor={sc}  SigmaSpace={ss}  |  'f' fullscreen",
                "[G]eometric   [C]olor   [B]right/Contrast   [H]istogram   [K] Gaussian   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   [V]iew   ESC: quit",
                "0: reset (d=9, SigmaColor=75, SigmaSpace=75)   ·   'i' to exit"
            ])

        elif mode == MODE_CANNY:
            lower = get_trackbar_value('Lower', CANNY_TRACKBARS)
            upper = get_trackbar_value('Upper', CANNY_TRACKBARS)
            apidx = get_trackbar_value('ApertureIdx', CANNY_TRACKBARS)
            l2g   = get_trackbar_value('L2Grad', CANNY_TRACKBARS)
            pblur = max(1, _ensure_odd(get_trackbar_value('PreBlur', CANNY_TRACKBARS)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if pblur > 1:
                gray = cv2.GaussianBlur(gray, (pblur, pblur), 0)

            aperture = [3, 5, 7][int(np.clip(apidx, 0, 2))]
            edges = cv2.Canny(gray, threshold1=lower, threshold2=upper, apertureSize=aperture, L2gradient=bool(l2g))
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            draw_bottom_menu(display, [
                f"Mode: Canny Edge  |  Lower={lower}  Upper={upper}  Aperture={aperture}  L2Grad={'ON' if l2g else 'OFF'}  PreBlur={pblur}  |  'f' fullscreen",
                "[G]eometric   [C]olor   [B]right/Contrast   [H]istogram   [K] Gaussian   [I] Bilateral   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   [V]iew   ESC: quit",
                "Press '0' to reset (Lower=50, Upper=150, Aperture=3, L2Grad=0, PreBlur=1)   ·   'e' to exit"
            ])

        elif mode == MODE_HOUGH:
            lower = get_trackbar_value('HLower', HOUGH_TRACKBARS)
            upper = get_trackbar_value('HUpper', HOUGH_TRACKBARS)
            apidx = get_trackbar_value('HApertureId', HOUGH_TRACKBARS)
            l2g   = get_trackbar_value('HL2Grad', HOUGH_TRACKBARS)
            pblur = max(1, _ensure_odd(get_trackbar_value('HPreBlur', HOUGH_TRACKBARS)))
            use_p   = get_trackbar_value('UseP', HOUGH_TRACKBARS)
            rho     = max(1, get_trackbar_value('Rho', HOUGH_TRACKBARS))
            tstep_d = max(1, get_trackbar_value('ThetaDeg', HOUGH_TRACKBARS))
            theta   = np.deg2rad(tstep_d)
            thresh  = max(1, get_trackbar_value('Thresh', HOUGH_TRACKBARS))
            minlen  = get_trackbar_value('MinLen', HOUGH_TRACKBARS)
            maxgap  = get_trackbar_value('MaxGap', HOUGH_TRACKBARS)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if pblur > 1:
                gray = cv2.GaussianBlur(gray, (pblur, pblur), 0)
            aperture = [3, 5, 7][int(np.clip(apidx, 0, 2))]
            edges = cv2.Canny(gray, threshold1=lower, threshold2=upper, apertureSize=aperture, L2gradient=bool(l2g))

            overlay = frame.copy()
            if use_p:
                linesP = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=thresh,
                                         minLineLength=max(0, minlen), maxLineGap=max(0, maxgap))
                count = 0 if linesP is None else len(linesP)
                if linesP is not None:
                    for x1,y1,x2,y2 in linesP[:,0,:]:
                        cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
                mode_text = f"HoughP  |  lines:{count}"
            else:
                lines = cv2.HoughLines(edges, rho=rho, theta=theta, threshold=thresh)
                count = _draw_standard_hough_lines(overlay, lines)
                mode_text = f"Hough (standard)  |  lines:{count}"

            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            gap = 8
            h, w = overlay.shape[:2]
            canvas = np.zeros((h, w*2 + gap, 3), dtype=np.uint8)
            canvas[:, :w] = overlay
            canvas[:, w+gap:w*2+gap] = edges_bgr
            display = canvas

            draw_bottom_menu(display, [
                f"Mode: {mode_text}  |  'f' fullscreen",
                f"Canny: Lower={lower} Upper={upper} Ap={aperture} L2={'ON' if l2g else 'OFF'} PreBlur={pblur}",
                f"Hough: UseP={use_p} Rho={rho} ThetaStep={tstep_d}° Thresh={thresh} MinLen={minlen} MaxGap={maxgap}",
                "[G]eometric   [C]olor   [B]right/Contrast   [K] Gaussian   [I] Bilateral   [E] Canny   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   [V]iew   ESC: quit",
                "Press '0' to reset Hough+Canny params   ·   't' to exit"
            ])

        elif mode == MODE_PANO:
            overlap = get_trackbar_value('Overlap', PANO_TRACKBARS)
            pano.set_overlap(overlap)

            canvas = pano.canvas
            if canvas is None:
                preview = frame.copy()
            else:
                h = max(canvas.shape[0], frame.shape[0])
                w_left = canvas.shape[1]
                w_right = frame.shape[1]
                gap = 10
                preview = np.zeros((h, w_left + gap + w_right, 3), dtype=np.uint8)
                preview[:canvas.shape[0], :canvas.shape[1]] = canvas
                preview[:frame.shape[0], w_left+gap:w_left+gap+w_right] = frame

            display = preview
            draw_bottom_menu(display, [
                "Mode: Panorama (L→R strip stitch)  |  'f' fullscreen",
                "[N] add frame   [R] reset   [S] save panorama.png (returns to View)",
                f"Status: {pano.last_status if pano.last_status else '—'}",
                f"Overlap: {overlap}px  (try 60–120px)",
                "[G]eometric   [C]olor   [B]right/Contrast   [H]istogram   [K] Gaussian   [I] Bilateral   [E] Canny   [T] Hough   [A]R   [K] Calib   [U] Undistort   [V]iew   ESC: quit",
                "Tip: rotate in place; 40–60% overlap; keep level."
            ])

        elif mode == MODE_AR:
            display = _draw_ar_wireframe(
                frame, aruco_kind, aruco_detector, aruco_dict, aruco_params,
                model_vertices, model_faces, cam_mtx, dist, marker_size_m=0.05
            )
            draw_bottom_menu(display, [
                "Mode: AR (OBJ wireframe)  |  'f' fullscreen",
                "[G]eometric   [C]olor   [B]right/Contrast   [H]istogram   [K] Gaussian   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [V]iew   [K] Calib   [U] Undistort   ESC: quit",
                "Tip: print a 6x6_250 ArUco marker; ensure calibration.npz exists (press K to create)."
            ])

        elif mode == MODE_CALIB:
            display = _run_calibration_step(frame, objpoints, imgpoints, last_capture_time, TARGET_IMAGES)
            draw_bottom_menu(display, [
                "Mode: Calibration (Chessboard)  |  'f' fullscreen",
                "Board: 9x6 inner corners, square=25mm. Hold steady; captures every 2s.",
                f"Captured: {len(objpoints)}/{TARGET_IMAGES}",
                "[K] to exit (or auto-continue when enough)."
            ])
            if len(objpoints) >= TARGET_IMAGES:
                calib_results = _finalize_calibration(objpoints, imgpoints, frame.shape[:2])
                cam_mtx, dist = calib_results['mtx'], calib_results['dist']
                objpoints.clear(); imgpoints.clear()
                mode = MODE_CALIB_RES

        elif mode == MODE_CALIB_RES:
            display = _show_calibration_summary(frame, calib_results)
            draw_bottom_menu(display, [
                "Mode: Calibration Result  |  'f' fullscreen",
                "[U] Undistort live   [V] View   [A] AR   [K] Re-calibrate"
            ])

        elif mode == MODE_UNDISTORT:
            display = _apply_undistort(frame, cam_mtx, dist)
            draw_bottom_menu(display, [
                "Mode: Undistort (live)  |  'f' fullscreen",
                "[V] View   [A] AR   [K] Calibrate"
            ])

        else:  # MODE_HIST
            mode_str = "gray" if hist_gray_only else "color"
            hist_canvas = make_hist_canvas(frame, mode=mode_str, log_scale=hist_log)
            preview_h = hist_canvas.shape[0]
            scale = preview_h / frame.shape[0]
            preview_w = int(frame.shape[1] * scale)
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

            gap = 12
            total_w = preview.shape[1] + gap + hist_canvas.shape[1]
            canvas = np.zeros((preview_h, max(total_w, 1), 3), dtype=np.uint8)
            canvas[:, :preview.shape[1]] = preview
            canvas[:, preview.shape[1] + gap : preview.shape[1] + gap + hist_canvas.shape[1]] = hist_canvas

            display = canvas
            draw_bottom_menu(display, [
                f"Mode: Histogram · {'GRAY' if hist_gray_only else 'BGR'} {'(log)' if hist_log else ''}  |  'f' fullscreen",
                "[G]eometric   [C]olor   [B]right/Contrast   [V]iew   [K] Gaussian   [I] Bilateral   [E] Canny   [T] Hough   [Y] Panorama   [A]R   [K] Calib   [U] Undistort   ESC: quit",
                "1: BGR histogram   2: GRAY histogram   3/L: toggle log scale"
            ], origin=(20, 40))

        cv2.imshow(WINDOW_NAME, display)

        # ── KEY HANDLING ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen: _apply_fullscreen()
            else: _exit_fullscreen()

        elif key == ord('v'):
            destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('g'):
            if mode != MODE_GEOMETRIC:
                setup_geometric_ui(fullscreen=fullscreen); mode = MODE_GEOMETRIC
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('c'):
            if mode != MODE_COLOR:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_COLOR
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('b'):
            if mode != MODE_TONE:
                setup_tone_ui(fullscreen=fullscreen); mode = MODE_TONE
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('h'):
            if mode != MODE_HIST:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_HIST
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('k'):
            if mode != MODE_CALIB:
                objpoints.clear(); imgpoints.clear()
                last_capture_time[0] = time.time()
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_CALIB
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('u'):
            if mode != MODE_UNDISTORT:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_UNDISTORT
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('i'):
            if mode != MODE_BILATERAL:
                setup_bilateral_ui(fullscreen=fullscreen); mode = MODE_BILATERAL
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('e'):
            if mode != MODE_CANNY:
                setup_canny_ui(fullscreen=fullscreen); mode = MODE_CANNY
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('t'):
            if mode != MODE_HOUGH:
                setup_hough_ui(fullscreen=fullscreen); mode = MODE_HOUGH
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('y'):
            if mode != MODE_PANO:
                setup_pano_ui(fullscreen=fullscreen); mode = MODE_PANO
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        elif key == ord('a') and mode != MODE_GEOMETRIC:
            if mode != MODE_AR:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_AR
            else:
                destroy_window(); create_window(fullscreen=fullscreen); mode = MODE_VIEW

        # Geometric-only keys
        elif mode == MODE_GEOMETRIC:
            if key == ord('m'):
                submode = SUB_SIMILARITY if submode == SUB_RIGID else SUB_RIGID
            elif key == ord(' '):
                auto_spin = not auto_spin
            elif key == ord('a'):
                ang = get_trackbar_value('Angle', GEOM_TRACKBARS)
                set_trackbar_value('Angle', int(round(wrap_angle_deg(ang - 5))), GEOM_TRACKBARS)
            elif key == ord('d'):
                ang = get_trackbar_value('Angle', GEOM_TRACKBARS)
                set_trackbar_value('Angle', int(round(wrap_angle_deg(ang + 5))), GEOM_TRACKBARS)
            elif key == ord('0'):
                auto_spin = False
                reset_geometric_ui()

        # Tone-only keys
        elif mode == MODE_TONE and key == ord('0'):
            set_trackbar_value('Brightness', 0,   TONE_TRACKBARS)
            set_trackbar_value('Contrast',   100, TONE_TRACKBARS)

        # Color-only keys
        elif mode == MODE_COLOR:
            if key == ord('1'): color_mode = COLOR_RGB
            elif key == ord('2'): color_mode = COLOR_GRAY
            elif key == ord('3'): color_mode = COLOR_HSV

        # Blur-only keys
        elif mode == MODE_BLUR and key == ord('0'):
            set_trackbar_value('Kernel', 5,   BLUR_TRACKBARS)
            set_trackbar_value('SigmaX', 10,  BLUR_TRACKBARS)

        # Bilateral-only keys
        elif mode == MODE_BILATERAL and key == ord('0'):
            set_trackbar_value('Diameter',   9,  BILAT_TRACKBARS)
            set_trackbar_value('SigmaColor', 75, BILAT_TRACKBARS)
            set_trackbar_value('SigmaSpace', 75, BILAT_TRACKBARS)

        # Canny-only keys
        elif mode == MODE_CANNY and key == ord('0'):
            reset_canny_ui()

        # Hough-only keys
        elif mode == MODE_HOUGH and key == ord('0'):
            reset_hough_ui()

        # Panorama-only keys
        elif mode == MODE_PANO:
            if key == ord('n'):
                pano.add_image(frame)
            elif key == ord('r'):
                pano.reset()
            elif key == ord('s'):
                canvas = pano.canvas
                if canvas is not None:
                    cv2.imwrite('assets/panorama.png', canvas)
                    pano.last_status = "Saved panorama.png"
                    destroy_window()
                    create_window(fullscreen=fullscreen)
                    mode = MODE_VIEW

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
