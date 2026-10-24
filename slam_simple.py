"""
Feature tracking only, fed by phone_stream.py. No pose estimation, no
mapping - that all moves to OpenVINS (see SPEC_openvins_floorplan.md).

Detects ORB features per frame and matches them against the previous
frame, drawing keypoints and their frame-to-frame motion over the live
view. Useful for judging whether the camera feed is good enough to track
on at all.

The matched features also give a DIRECTION of travel: the pattern they
form grows when the camera moves toward what it is looking at and slides
when it moves across, once the gyro's rotation has been taken out. That
direction is drawn as an arrow in the same 3D space as the phone box, so
moving the way the lens points lines the arrow up with the phone. It is a
direction only - monocular flow cannot give distance.

    ./venv/bin/python slam_simple.py
"""

import time
from collections import deque

import cv2
import numpy as np

from phone_stream import PhoneStream

IMU_STRIP_H = 150  # height of the black readout strip under the image
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Rectangular prism roughly phone-shaped: thin in Z, tall in Y. Device
# axes are X across the screen, Y up the screen, Z out of the screen.
_BOX_HALF = np.array([0.30, 0.60, 0.05])
_BOX_VERTS = np.array([[sx, sy, sz] for sz in (-1, 1)
                       for sy in (-1, 1) for sx in (-1, 1)], float) * _BOX_HALF

# (vertex indices, BGR). Colours make the pose readable at a glance:
# pale blue is the screen, dark is the back where the camera is.
_BOX_FACES = [
    ((0, 1, 3, 2), (70, 70, 70)),      # -Z back / camera side
    ((4, 5, 7, 6), (235, 195, 125)),   # +Z screen
    ((0, 1, 5, 4), (105, 105, 105)),   # -Y bottom
    ((2, 3, 7, 6), (165, 165, 165)),   # +Y top
    ((0, 2, 6, 4), (130, 130, 130)),   # -X left
    ((1, 3, 7, 5), (130, 130, 130)),   # +X right
]


def _view_matrix(az_deg=35.0, el_deg=25.0):
    """World -> view. Looks down from above at an angle so all three axes read."""
    rz = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(az_deg)]))[0]
    rx = cv2.Rodrigues(np.array([np.radians(el_deg - 90.0), 0.0, 0.0]))[0]
    return rx @ rz


_VIEW = _view_matrix()

# Phone held perfectly upright, camera pointing at heading 0. Pitch and
# roll are locked to this; only yaw moves.
_UPRIGHT = cv2.Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))[0]

# Compass readings are jumpy; low-pass so the box does not jitter.
# Higher tracks faster but shakes more.
HEADING_SMOOTHING = 0.15

# Tipping the phone back toward lens-up swings the reported heading by
# 180. Upright, the only device axis lying horizontally is the camera
# axis, so that is what the heading follows; tipped flat it is the top
# edge instead. Those two agree at lens-DOWN but are opposite at lens-UP,
# hence a flip on that side only.
#
# The switch is DETECTED rather than predicted from a pitch angle: a
# fixed threshold has to guess where iOS changes over, and anywhere the
# guess differs the box sits 180 out until the threshold catches up. The
# switch itself is unmistakable - a ~180 change within a single reading,
# which no real turn can produce (even a 360 deg/s spin moves the compass
# only a few degrees between readings). Gated on pitch as a sanity check,
# since near upright a jump that size is a magnetic glitch, not a switch.
COMPASS_FLIP_DETECT = 120.0
COMPASS_FLIP_MIN_PITCH = 30.0

# Very near straight up or down, no device axis is meaningfully
# horizontal and the heading is genuinely undefined, so hold it there.
COMPASS_MAX_PITCH = 80.0

# A single reading that disagrees this much is treated as a glitch, not
# motion - at any believable turn rate the change between readings is far
# smaller. Held only until enough consecutive readings agree, so a
# genuine fast spin still gets through rather than sticking forever.
COMPASS_JUMP_DEG = 50.0
COMPASS_JUMP_CONFIRM = 4


def upright_attitude(compass_deg, pitch_deg=0.0):
    """
    Device -> world. Roll stays locked; pitch tips the phone forward about
    its own X axis, so 0 is upright and 90 is flat with the camera looking
    down. Yaw then swings the whole thing to the compass heading.
    """
    yaw = cv2.Rodrigues(np.array([0.0, 0.0, -np.radians(compass_deg)]))[0]
    pitch = cv2.Rodrigues(np.array([-np.radians(pitch_deg), 0.0, 0.0]))[0]
    return yaw @ _UPRIGHT @ pitch


def draw_phone_box(canvas, R, centre, scale):
    """
    Draw the device as a solid prism under rotation R (device -> world).
    Faces are painter-sorted back to front, which is exact for a convex
    solid and avoids needing a depth buffer.
    """
    view = (_VIEW @ (R @ _BOX_VERTS.T)).T
    pts = np.stack([centre[0] + scale * view[:, 0],
                    centre[1] - scale * view[:, 1]], axis=1).astype(np.int32)

    for idx, colour in sorted(_BOX_FACES, key=lambda f: view[list(f[0]), 2].mean()):
        cv2.fillConvexPoly(canvas, pts[list(idx)], colour, cv2.LINE_AA)
        cv2.polylines(canvas, [pts[list(idx)]], True, (25, 25, 25), 1, cv2.LINE_AA)

    # Lens dot on the back face, so which way the camera looks is obvious
    # - but only when that face is actually toward the viewer.
    back = (_VIEW @ (R @ np.array([0.0, 0.0, -1.0])))
    if back[2] > 0:
        lens = (_VIEW @ (R @ np.array([0.0, 0.35, -_BOX_HALF[2]])))
        cv2.circle(canvas, (int(centre[0] + scale * lens[0]),
                            int(centre[1] - scale * lens[1])),
                   max(2, int(scale * 0.05)), (20, 20, 20), -1, cv2.LINE_AA)


# --- Direction of travel from the feature flow ------------------------------
#
# A camera moving by t through a scene at depth Z sees a feature at image
# offset (x, y) from the principal point move by
#
#     dx = (x * tz - f * tx) / Z        dy = (y * tz - f * ty) / Z
#
# Read that as two separate effects on the whole pattern of features:
#
#   * the (x * tz, y * tz) part is a uniform SCALE about the centre - the
#     shape drawn by connecting features GROWS when moving toward it,
#   * the (-f * tx, -f * ty) part is a uniform SHIFT - the shape SLIDES
#     when moving perpendicular to where the camera looks.
#
# So fitting a scale-plus-shift model p1 = s * p0 + b to matched features
# inverts straight back to the direction of travel:
#
#     tz / Z = s - 1        tx / Z = -bx / f        ty / Z = -by / f
#
# The leftover common 1 / Z factor is the usual monocular scale ambiguity:
# this recovers WHICH WAY the camera moved, never how far. Walking 1m in a
# big room and 10cm in a small one look identical.
#
# Rotation has to be removed before fitting, because turning also shifts
# every feature sideways - a yaw and a strafe produce the same flow. That
# ambiguity is exactly what the gyro resolves, so the IMU is doing real
# work here rather than just driving the display.

# Rear-camera field of view, given as the DIAGONAL angle. A horizontal
# figure would be wrong here: the usual ~69 deg quoted for an iPhone main
# camera is across the sensor's LONG axis, but Safari delivers portrait, so
# the frame's width is the SHORT axis and spans a much narrower angle.
# Taking it diagonally makes the focal length come out right whichever way
# round the frame arrives. 81 deg reproduces the 69 deg horizontal spec on
# a 4:3 sensor.
#
# Focal length sets how sideways motion trades off against forward motion:
# tx/Z is -bx/f, so f reading 25% low tilts a 45 deg diagonal move by about
# 8 deg toward sideways. Worth getting roughly right, not worth calibrating.
DIAGONAL_FOV_DEG = 81.0

MIN_FLOW_MATCHES = 12
FLOW_TRIM_ITERS = 2

# Scale is only separable from shift when the features span a range of
# radii - a tight cluster is explained equally well by zooming or sliding.
# Minimum radial spread of the inliers, as a fraction of the frame
# diagonal, before the forward/backward term is believed at all.
MIN_RADIAL_SPREAD = 0.08

# Above this turn rate the fit degrades two ways: gyro and frame-timing
# error start to dominate what survives de-rotation, and the scale-plus-
# shift model is only first order, so a large rotation within one frame
# interval biases the direction. At this limit (~5.7 deg per frame at
# 15fps) the synthetic error stays within a few degrees.
FLOW_MAX_RATE = 1.5       # rad/s

# Shift applied to frame capture times before looking up the gyro, to undo
# any constant lag in the camera pipeline. Cannot be derived - it is a
# property of this phone - so it is measured instead: the readout shows the
# offset that best cancels rotation ("t-off"), and once that settles on a
# consistent value, set it here.
CAM_IMU_OFFSET_S = 0.0

OFFSET_SEARCH_S = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
OFFSET_MIN_RATE = 0.5     # rad/s below which the estimate is meaningless
OFFSET_SMOOTHING = 0.05   # heavy: this is a fixed property, not a signal
OFFSET_MIN_SAMPLES = 30   # updates to collect before trusting it enough to
                          # correct with - at 0.05 smoothing that is roughly
                          # where it has converged

# Per second, in units of scene depth: 0.03 is ~9cm/s at 3m away.
FLOW_DEADBAND = 0.03
MOTION_SMOOTHING = 0.25   # EMA weight on each new direction estimate
MOTION_DECAY = 0.85       # per frame, with nothing trustworthy to add
ARROW_FULL_SCALE = 0.35   # speed drawn at full arrow length (~1m/s at 3m)
ARROW_MIN_DRAW = 0.02

# Camera axes (OpenCV: +x image right, +y image DOWN, +z out of the lens)
# -> device axes (+x screen right, +y screen up, +z out of the screen, so
# the rear lens looks along -z). Safari delivers the frame already rotated
# to portrait, so image axes line up with screen axes and this is just a
# sign flip on two of them.
#
# To check it: PAN or TILT the phone and watch "rot" in the readout, the
# median flow before and after de-rotation. A correct mapping collapses it
# to near zero; one a quarter turn out leaves it HIGHER than it started and
# invents a large sideways motion. Rolling about the lens axis cannot tell
# the two apart, so do not test with that.
CAM_TO_DEV = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])


def focal_px(frame_shape):
    """Focal length in pixels for this frame, from the diagonal FOV."""
    h, w = frame_shape[:2]
    return (np.hypot(w, h) / 2.0) / np.tan(np.radians(DIAGONAL_FOV_DEG / 2.0))


def fit_scale_shift(q0, q1):
    """
    Least-squares fit of q1 = s * q0 + b over centred pixel coordinates,
    with a couple of rounds of outlier trimming - ORB matches always
    include some wrong pairings, and one bad match placed far from the
    centre drags the scale term hard.

    Returns (s, bx, by, keep_mask) or None if too little survives.
    """
    keep = np.ones(len(q0), bool)
    for _ in range(FLOW_TRIM_ITERS + 1):
        if keep.sum() < MIN_FLOW_MATCHES:
            return None
        a0, a1 = q0[keep], q1[keep]
        n = len(a0)

        # Two rows per match: x and y share s but have their own shift.
        A = np.zeros((2 * n, 3))
        A[0::2, 0], A[0::2, 1] = a0[:, 0], 1.0
        A[1::2, 0], A[1::2, 2] = a0[:, 1], 1.0
        y = np.empty(2 * n)
        y[0::2], y[1::2] = a1[:, 0], a1[:, 1]

        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        s, b = float(sol[0]), sol[1:]

        residual = np.linalg.norm(q1 - (q0 * s + b), axis=1)
        # Scaled off the median so the cut adapts to how much the camera
        # actually moved; the floor keeps a near-still frame from trimming
        # almost everything as an "outlier".
        keep = residual < max(2.0, 2.5 * float(np.median(residual[keep])))

    return s, float(b[0]), float(b[1]), keep


def derotate(p0, R_cam, focal, centre):
    """
    Where each feature would land if the camera had only rotated. Unproject
    to a ray, turn it by R_cam, project back. Returns (points, mask) - the
    mask drops anything that rotation put behind the camera.
    """
    rays = np.column_stack([(p0 - centre) / focal, np.ones(len(p0))])
    rot = rays @ R_cam.T
    ahead = rot[:, 2] > 1e-6
    return centre + focal * rot[ahead, :2] / rot[ahead, 2:3], ahead


def estimate_time_offset(stream, t0, t1, p0, p1, focal, centre):
    """
    Which camera/gyro time offset best cancels the rotation flow.

    A frame's capture timestamp and the gyro's timestamps come from the
    same clock, but the camera pipeline may report a time that does not
    match when the sensor was actually exposed. A constant lag like that
    leaves rotation behind that de-rotation cannot remove, in proportion to
    how fast the phone is turning - which the flow fit then reports as
    sideways translation.

    So: scan a few offsets, take the one leaving the least flow, and fit a
    parabola through its neighbours so the answer is not stuck on the step
    size. Only means anything while actually turning - with no rotation,
    every offset de-rotates equally well.
    """
    res = []
    for off in OFFSET_SEARCH_S:
        R_dev, _, ok = stream.get_rotation(t0 + off, t1 + off)
        if not ok:
            return None
        p0r, ahead = derotate(p0, CAM_TO_DEV.T @ R_dev.T @ CAM_TO_DEV,
                              focal, centre)
        if ahead.sum() < MIN_FLOW_MATCHES:
            return None
        res.append(float(np.median(np.linalg.norm(p1[ahead] - p0r, axis=1))))

    k = int(np.argmin(res))
    if k == 0 or k == len(res) - 1:
        return OFFSET_SEARCH_S[k]        # best is at the edge of the scan
    a, b, cc = res[k - 1], res[k], res[k + 1]
    denom = a - 2.0 * b + cc
    shift = 0.5 * (a - cc) / denom if abs(denom) > 1e-9 else 0.0
    step = OFFSET_SEARCH_S[k + 1] - OFFSET_SEARCH_S[k]
    return OFFSET_SEARCH_S[k] + float(np.clip(shift, -1.0, 1.0)) * step


def flow_translation(p0, p1, R_cam, focal, centre, diag):
    """
    Direction of travel from matched features p0 -> p1, in the CURRENT
    frame's camera axes - not the previous frame's. The fit is done against
    de-rotated coordinates, which already live in the new frame, so the
    result pairs with the current attitude and must not be rotated again.
    Checked synthetically: mixing the two shows up as an error equal to the
    rotation over the interval.

    R_cam carries a ray from the previous frame's camera axes into this
    one's, so applying it to p0 predicts where each feature would have
    landed from rotation alone. Whatever flow is left over is translation.

    Returns (t_cam, n_inliers, raw_px, derot_px) or None. t_cam is t / Z
    for this frame interval: a direction, with magnitude in depth units.
    """
    p0r, ahead = derotate(p0, R_cam, focal, centre)
    if ahead.sum() < MIN_FLOW_MATCHES:
        return None
    p1a, p0a = p1[ahead], p0[ahead]

    # How much flow the rotation accounted for. If de-rotation is working,
    # derot_px is well below raw_px while panning; if the camera-to-device
    # mapping is wrong it will be no better, or worse.
    raw_px = float(np.median(np.linalg.norm(p1a - p0a, axis=1)))
    derot_px = float(np.median(np.linalg.norm(p1a - p0r, axis=1)))

    fit = fit_scale_shift(p0r - centre, p1a - centre)
    if fit is None:
        return None
    s, bx, by, keep = fit

    # Forward motion is the weakest thing in the whole estimate, so drop it
    # rather than report a scale fitted from a cluster of features that
    # cannot distinguish zoom from slide.
    spread = float(np.linalg.norm(p0r[keep] - centre, axis=1).std())
    if spread < MIN_RADIAL_SPREAD * diag:
        s = 1.0

    t_cam = np.array([-bx / focal, -by / focal, s - 1.0])
    return t_cam, int(keep.sum()), raw_px, derot_px


def draw_motion_arrow(canvas, vec_world, centre, max_len):
    """
    Arrow from the box's centre along the world-frame direction of travel,
    drawn in the same projection as the box so the two read together: move
    the way the lens points and the arrow lines up with the phone.
    """
    speed = float(np.linalg.norm(vec_world))
    if speed < ARROW_MIN_DRAW:
        return
    unit = vec_world / speed
    length = max_len * min(1.0, speed / ARROW_FULL_SCALE)

    def project(v):
        p = _VIEW @ v
        return (int(centre[0] + length * p[0]), int(centre[1] - length * p[1]))

    origin = (int(centre[0]), int(centre[1]))

    # Horizontal shadow first: without it an arrow pointing away from the
    # viewer is hard to tell from one pointing down.
    flat = unit * np.array([1.0, 1.0, 0.0])
    if np.linalg.norm(flat) > 1e-3:
        cv2.line(canvas, origin, project(flat), (40, 110, 40), 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, origin, project(unit), (90, 255, 90), 2,
                    cv2.LINE_AA, tipLength=0.3)


# --- Path, in plan view -----------------------------------------------------
#
# The flow estimate is t / Z: a direction, with its length in units of
# however far away the scene is. Summing those steps gives a path, but only
# after multiplying by a depth to turn them into metres - and monocular
# vision cannot supply one. So it is assumed here, flat.
#
# What that costs: a guess that is uniformly wrong only resizes the path,
# leaving its shape alone. The real damage is that the depth in view
# CHANGES as you walk, so each stretch is scaled differently. Simulated
# walking a 4x3m loop in an 8x6m room, the four legs came back scaled
# 0.51x, 0.67x, 1.07x and 1.13x - the turns and their directions were all
# correct, but the rectangle's proportions were not.
#
# Note it is not the average depth in view that sets a leg's scale. The fit
# weights points by radius from the centre, so the edges of the frame
# dominate, and those are usually the near side walls rather than whatever
# is far away straight ahead.
#
# Heading is the part that holds up. Pure visual odometry accumulates yaw
# error and spirals; here yaw is pinned to the compass, so the path's
# directions stay honest even where its distances do not.
#
# Adjustable with [ and ] in the window, which also clears the path - walk
# a known distance and set it so the plan agrees.
ASSUMED_DEPTH_M = 3.0

PLAN_PANEL_W = 300
PLAN_MIN_SPAN_M = 1.5     # never zoom in closer than this, so early noise
                          # does not fill the panel
PATH_MAX_POINTS = 20000
PATH_MIN_STEP_M = 0.01    # do not store steps smaller than this


def draw_plan_view(canvas, path, heading_deg, span, depth_m):
    """
    Top-down plot of where the camera has been. World X is drawn right and
    world Y up, so the plan reads like a map with north up - which is
    meaningful here only because yaw comes from the compass.
    """
    h, w = canvas.shape[:2]
    canvas[:] = (16, 16, 16)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (55, 55, 55), 1)
    cv2.putText(canvas, "plan view - north up", (8, 20), FONT, 0.45,
                (170, 170, 170), 1)
    cv2.putText(canvas, f"assumes scene {depth_m:.1f}m away", (8, 38), FONT,
                0.38, (110, 110, 110), 1)

    if len(path) < 2:
        cv2.putText(canvas, "waiting for motion...", (8, h // 2), FONT, 0.45,
                    (110, 110, 110), 1)
        return

    pts = np.array(path)[:, :2]
    mid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    margin = 34
    px_per_m = (min(w, h) - 2 * margin) / span

    def to_px(p):
        return (int(w / 2 + (p[0] - mid[0]) * px_per_m),
                int(h / 2 - (p[1] - mid[1]) * px_per_m))

    poly = np.array([to_px(p) for p in pts], np.int32)
    cv2.polylines(canvas, [poly], False, (90, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[0]), 4, (130, 130, 130), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(poly[-1]), 5, (90, 255, 90), -1, cv2.LINE_AA)

    # Where the lens is pointing now, at the current position. Heading is
    # clockwise from north, so it is (sin, cos) in world X/Y.
    a = np.radians(heading_deg)
    cv2.arrowedLine(canvas, tuple(poly[-1]),
                    (int(poly[-1][0] + 18 * np.sin(a)),
                     int(poly[-1][1] - 18 * np.cos(a))),
                    (90, 255, 90), 1, cv2.LINE_AA, tipLength=0.4)

    # Scale bar at a round number of metres - the only honest way to show
    # distance when the zoom keeps changing.
    bar_m = 50.0
    for v in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        bar_m = v
        if v * px_per_m > 40:
            break
    y0 = h - 14
    cv2.line(canvas, (12, y0), (12 + int(bar_m * px_per_m), y0),
             (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{bar_m:g}m", (12, y0 - 6), FONT, 0.4,
                (200, 200, 200), 1)

    walked = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    cv2.putText(canvas, f"walked {walked:.1f}m   net {np.linalg.norm(pts[-1] - pts[0]):.1f}m",
                (8, h - 32), FONT, 0.4, (140, 140, 140), 1)


def main():
    # Starts the HTTPS + WebSocket server and prints the URL to open in
    # Safari on the phone. Frames then arrive on a background thread; we
    # pull the most recent one with get_latest_frame() below.
    stream = PhoneStream()
    stream.start()

    # ORB finds corner-like points and computes a 256-bit binary
    # descriptor for each, so the same physical point can be recognised in
    # a later frame. nfeatures caps how many are kept per frame - higher
    # tracks better but costs more in both detection and matching.
    orb = cv2.ORB_create(nfeatures=500)

    # NORM_HAMMING is the correct metric for ORB's binary descriptors
    # (count the differing bits). crossCheck keeps only mutual best
    # matches - A's best match is B and B's best match is A - which
    # discards a lot of bad pairings for free.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Previous frame's features, so each new frame has something to match
    # against. Populated at the bottom of the loop.
    prev_kp, prev_des = None, None
    prev_capture = None
    frame_count = 0
    motion_ema = np.zeros(3)   # world-frame direction of travel, smoothed
    position = np.zeros(3)     # running sum of the per-frame steps, metres
    path = deque([position.copy()], maxlen=PATH_MAX_POINTS)
    plan_span = PLAN_MIN_SPAN_M   # grows only, so the plan does not rescale
                                  # under your feet on every frame
    depth_m = ASSUMED_DEPTH_M
    offset_est = None          # measured camera/gyro lag, seconds
    offset_n = 0
    smoothed_heading = None
    pitch_now = 0.0
    jump_count = 0
    compass_flipped = False

    print("Waiting for the phone to connect and start streaming...")
    print("In the window: 'r' resets the path, '[' / ']' change the "
          "assumed depth, 'q' quits.")

    last_seen_frame_time = None
    while True:
        # get_latest_frame() returns whatever arrived most recently, which
        # may be one we already processed. Comparing timestamps means we
        # only do work on genuinely new frames; the sleep stops this from
        # spinning the CPU at 100% while waiting for the next one.
        frame, frame_time, capture = stream.get_latest_frame()
        if frame is None or frame_time == last_seen_frame_time:
            time.sleep(0.005)
            continue
        if last_seen_frame_time is None:
            h0, w0 = frame.shape[:2]
            print(f"first frame {w0}x{h0}, focal ~{focal_px(frame.shape):.0f}px")
            if w0 > h0:
                print("  WARNING: landscape frame. CAM_TO_DEV assumes Safari's "
                      "portrait delivery, so the\n"
                      "  sideways/vertical split will be a quarter turn out. "
                      "Forward/back is unaffected.")
        last_seen_frame_time = frame_time

        # Direction of travel for THIS frame pair, in camera axes, or None
        # when it could not be trusted - too few matches, too fast a turn,
        # or no gyro samples covering the interval. Both reset every frame
        # so the readout can never show figures from an older pair.
        flow_rate_cam = None
        flow_disp_cam = None
        flow_text = None

        # --- Detect ------------------------------------------------------
        # ORB works on intensity, not colour, so convert first.
        # detectAndCompute returns two parallel results: kp (where each
        # feature is in the image) and des (what each one looks like, as
        # the binary descriptor used for matching).
        t_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        detect_ms = (time.time() - t_start) * 1000

        display = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))

        # --- Match against the previous frame ----------------------------
        n_matches = 0
        if prev_des is not None and des is not None:
            matches = matcher.match(prev_des, des)

            # NOTE: m.distance is the DESCRIPTOR distance (how similar the
            # two features look), not how far the point moved in pixels.
            # Sorting by it keeps the most confident matches; the cap is
            # for display readability.
            matches = sorted(matches, key=lambda m: m.distance)[:150]
            n_matches = len(matches)

            # The argument order above decides which index means what:
            # prev_des was passed first, so queryIdx indexes the PREVIOUS
            # frame and trainIdx the CURRENT one. Swapping these silently
            # reverses the direction of every motion line.
            for m in matches:
                x0, y0 = prev_kp[m.queryIdx].pt
                x1, y1 = kp[m.trainIdx].pt
                cv2.line(display, (int(x0), int(y0)), (int(x1), int(y1)),
                         (0, 200, 255), 1)

            # --- Direction of travel -------------------------------------
            # Same matches, read for motion instead of just drawn. Needs
            # the gyro rotation over exactly this frame interval, which is
            # why frames carry a capture time rather than an arrival time.
            dt = (capture - prev_capture
                  if capture is not None and prev_capture is not None else 0.0)
            if n_matches >= MIN_FLOW_MATCHES and 0.0 < dt < 0.5:
                # Self-calibrated once enough turning has been seen,
                # falling back to the hand-set constant before that. The
                # estimator below always works from the raw timestamps, so
                # applying this cannot feed back into it.
                applied_off = (offset_est
                               if offset_est is not None
                               and offset_n >= OFFSET_MIN_SAMPLES
                               else CAM_IMU_OFFSET_S)
                R_dev, gyro_rate, rot_ok = stream.get_rotation(
                    prev_capture + applied_off, capture + applied_off)

                # R_dev maps device-at-now into device-at-prev, so its
                # transpose carries the previous frame's rays forward into
                # this one; conjugating by C says the same thing in camera
                # axes.
                R_cam = CAM_TO_DEV.T @ R_dev.T @ CAM_TO_DEV

                h, w = frame.shape[:2]
                focal = focal_px(frame.shape)
                centre = np.array([w / 2.0, h / 2.0])
                p0 = np.array([prev_kp[m.queryIdx].pt for m in matches], float)
                p1 = np.array([kp[m.trainIdx].pt for m in matches], float)

                flow = flow_translation(p0, p1, R_cam, focal, centre,
                                        float(np.hypot(w, h)))
                if flow is not None:
                    flow_text = (flow[1], flow[2], flow[3], gyro_rate)
                    if rot_ok and gyro_rate < FLOW_MAX_RATE:
                        flow_rate_cam = flow[0] / dt   # per second
                        flow_disp_cam = flow[0]        # this interval

                # Only while turning: with the phone still, every offset
                # cancels the (absent) rotation equally and the minimum is
                # noise. Smoothed hard, since the true value is a constant.
                if rot_ok and gyro_rate > OFFSET_MIN_RATE:
                    best = estimate_time_offset(stream, prev_capture, capture,
                                                p0, p1, focal, centre)
                    if best is not None:
                        offset_est = (best if offset_est is None else
                                      offset_est + OFFSET_SMOOTHING
                                      * (best - offset_est))
                        offset_n += 1

        # --- IMU readout -------------------------------------------------
        # Drawn into a black strip added below the image rather than on
        # top of it, so the text stays legible over bright scenes.
        # Diagnostic only - nothing here feeds the tracking above.
        cam_h, cam_w = frame.shape[:2]
        display = cv2.hconcat([display,
                               np.zeros((cam_h, PLAN_PANEL_W, 3), np.uint8)])
        display = cv2.copyMakeBorder(display, 0, IMU_STRIP_H, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        imu = stream.get_latest_imu()
        y = frame.shape[0]
        if imu is None:
            cv2.putText(display, "no IMU data (motion permission denied?)",
                        (8, y + 24), FONT, 0.5, (0, 140, 255), 1)
        else:
            (ax, ay, az), (wx, wy, wz), rate = imu
            cv2.putText(display,
                        f"accel  x{ax:+6.2f} y{ay:+6.2f} z{az:+6.2f}  m/s2 (raw, incl. gravity)",
                        (8, y + 20), FONT, 0.45, (220, 220, 220), 1)
            cv2.putText(display,
                        f"gyro   x{wx:+6.2f} y{wy:+6.2f} z{wz:+6.2f}  rad/s",
                        (8, y + 40), FONT, 0.45, (220, 220, 220), 1)
            # VIO generally wants 200Hz+; the browser caps out around 60,
            # so flag anything well below that as a problem worth chasing.
            rate_colour = (120, 255, 120) if rate >= 40 else (0, 140, 255)
            cv2.putText(display, f"IMU {rate:5.1f} Hz", (8, y + 60),
                        FONT, 0.45, rate_colour, 1)

            # Gyro-tracked, gravity-levelled. Signed, so camera-up and
            # camera-down are distinguishable. Roll is still locked.
            _, _, pitch, ready = stream.get_orientation()
            if ready:
                pitch_now = pitch
                cv2.putText(display,
                            f"pitch {pitch_now:+6.1f} deg  (+90 cam down, -90 cam up)",
                            (8, y + 82), FONT, 0.45, (255, 210, 120), 1)
            else:
                cv2.putText(display, "pitch  levelling...", (8, y + 82),
                            FONT, 0.45, (0, 140, 255), 1)

        # Compass is the one absolute yaw reference - gravity cannot give
        # it. Shown separately from 'facing' so the two can be compared:
        # they differ by a constant (wherever you started, relative to
        # north), and 'corr' means that gap is actively pinning yaw drift.
        compass = stream.get_compass()
        if compass is None:
            cv2.putText(display, "compass  no reading", (8, y + 104),
                        FONT, 0.45, (150, 150, 150), 1)
        else:
            c_heading, c_acc, correcting = compass
            acc_txt = "?" if c_acc is None else f"+-{c_acc:.0f}"
            state = "corr" if correcting else "idle"
            colour = (255, 210, 120) if correcting else (150, 150, 150)
            flip_txt = "  FLIP+180" if compass_flipped else ""
            cv2.putText(display,
                        f"compass {c_heading:6.1f} deg {acc_txt:>5}   {state}{flip_txt}",
                        (8, y + 104), FONT, 0.45, colour, 1)

        # 3D view, right of the text. Yaw from the compass, pitch from the
        # gravity-derived tilt. Roll stays locked.
        if compass is not None:
            raw_heading = compass[0]
            heading_in = (raw_heading + 180.0) % 360.0 if compass_flipped else raw_heading

            if smoothed_heading is None:
                smoothed_heading = heading_in
            else:
                # Shortest way round, so 359 -> 1 does not spin backwards.
                step = (heading_in - smoothed_heading + 180.0) % 360.0 - 180.0

                # A ~180 lurch while tipped is iOS changing which axis it
                # measures against. Follow it the moment it happens, so
                # there is no window where the box sits backwards. Toggling
                # makes the reading agree again, so it will not re-trigger.
                if (abs(step) > COMPASS_FLIP_DETECT
                        and abs(pitch_now) > COMPASS_FLIP_MIN_PITCH):
                    compass_flipped = not compass_flipped
                    heading_in = ((raw_heading + 180.0) % 360.0
                                  if compass_flipped else raw_heading)
                    step = (heading_in - smoothed_heading + 180.0) % 360.0 - 180.0

                if abs(pitch_now) > COMPASS_MAX_PITCH:
                    pass  # too flat for the heading to mean anything
                elif abs(step) > COMPASS_JUMP_DEG:
                    jump_count += 1
                    if jump_count >= COMPASS_JUMP_CONFIRM:
                        smoothed_heading = heading_in  # persisted, so real
                        jump_count = 0
                else:
                    jump_count = 0
                    smoothed_heading = (smoothed_heading + step * HEADING_SMOOTHING) % 360.0

        # --- Direction of travel, into world axes ------------------------
        # The flow estimate arrives in camera axes, which spin with the
        # phone. Rotating it into world axes first means the smoothing
        # averages a fixed physical direction instead of smearing it out
        # whenever the camera turns, and lets the steps simply be summed.
        R_att = (upright_attitude(smoothed_heading, pitch_now)
                 if smoothed_heading is not None else None)

        if flow_rate_cam is not None and R_att is not None:
            t_world = R_att @ (CAM_TO_DEV @ flow_rate_cam)
            if np.linalg.norm(t_world) > FLOW_DEADBAND:
                motion_ema += MOTION_SMOOTHING * (t_world - motion_ema)

                # The path integrates the RAW per-frame displacement, not
                # the smoothed rate: the EMA lags by design, which would
                # round off every corner. Summing raw steps is fine because
                # integration averages random error itself - only bias
                # accumulates. The deadband above is what stops a standing
                # still from random-walking the position around.
                position = position + depth_m * (
                    R_att @ (CAM_TO_DEV @ flow_disp_cam))
                if np.linalg.norm(position - path[-1]) > PATH_MIN_STEP_M:
                    path.append(position.copy())
            else:
                motion_ema *= MOTION_DECAY
        else:
            # Nothing trustworthy this frame, so let the arrow fall away
            # rather than hold a stale direction.
            motion_ema *= MOTION_DECAY

        # --- Plan view ---------------------------------------------------
        if len(path) > 1:
            pts = np.array(path)[:, :2]
            needed = float((pts.max(axis=0) - pts.min(axis=0)).max())
            plan_span = max(plan_span, needed * 1.1)
        plan = np.zeros((cam_h, PLAN_PANEL_W, 3), np.uint8)
        draw_plan_view(plan, path, smoothed_heading or 0.0, plan_span, depth_m)
        display[0:cam_h, cam_w:cam_w + PLAN_PANEL_W] = plan

        if R_att is not None:
            panel = min(160, display.shape[1] // 3)
            box_centre = (display.shape[1] - panel // 2 - 10,
                          y + IMU_STRIP_H // 2)
            draw_phone_box(display, R_att, box_centre, panel * 0.42)
            # Drawn over the box, from its centre, so the arrow and the
            # phone's facing can be compared directly.
            draw_motion_arrow(display, motion_ema, box_centre, panel * 0.46)

            # Reported back in device axes - what you can actually check by
            # moving the phone - rather than the world axes it is smoothed
            # in. 'rot' is the median flow before and after de-rotation: it
            # should drop sharply while panning, which is the check that the
            # gyro and the camera agree about which way is which.
            d = R_att.T @ motion_ema
            txt = f"move fwd{-d[2]:+5.2f} rt{d[0]:+5.2f} up{d[1]:+5.2f} /s"
            if flow_text is not None:
                n_in, raw_px, derot_px, rate = flow_text
                txt += f"  n={n_in:3d}  rot {raw_px:4.1f}>{derot_px:4.1f}px"
            if offset_est is not None:
                live = "*" if offset_n >= OFFSET_MIN_SAMPLES else "?"
                txt += f"  t-off{offset_est * 1000:+4.0f}ms{live}"
            cv2.putText(display, txt, (8, y + 126), FONT, 0.45,
                        (120, 255, 120), 1)

        # --- Display -----------------------------------------------------
        # waitKey also pumps the GUI event loop, so the window will not
        # repaint without it being called.
        cv2.imshow("Feature tracking", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Changing the depth rescales every future step, so mixing one
        # setting's steps with another's would be meaningless - clear the
        # path along with it.
        if key in (ord("r"), ord("["), ord("]")):
            if key == ord("["):
                depth_m = max(0.3, depth_m / 1.2)
            elif key == ord("]"):
                depth_m = min(50.0, depth_m * 1.2)
            if key != ord("r"):
                print(f"assumed depth now {depth_m:.2f}m (path cleared)")
            position = np.zeros(3)
            path = deque([position.copy()], maxlen=PATH_MAX_POINTS)
            plan_span = PLAN_MIN_SPAN_M

        # keypoints/matches show tracking quality; lag is how stale the
        # frame already was when we picked it up (network + decode time).
        frame_count += 1
        if frame_count % 10 == 0:
            lag_ms = (t_start - frame_time) * 1000
            print(f"frame {frame_count}: keypoints={len(kp)} matches={n_matches} "
                  f"timing(ms) lag={lag_ms:.0f} detect={detect_ms:.0f}")

        # This frame becomes the reference for the next one.
        prev_kp, prev_des = kp, des
        prev_capture = capture

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
