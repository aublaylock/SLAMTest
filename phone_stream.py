"""
phone_stream.py - serves a web page to the iPhone and receives its camera
+ IMU stream over a WebSocket.

Setup: run this file, open the printed https:// URL in Safari on the
phone, tap Start Streaming. If Tailscale HTTPS certs are enabled the cert
is automatic; otherwise a self-signed cert is generated and the printed
http:// URL serves it for one-time install + trust on the phone
(Settings > General > VPN & Device Management, then Settings > General >
About > Certificate Trust Settings).

Wire format (both ends are ours):
  binary: 8-byte big-endian float64 capture timestamp (ms) + JPEG frame
  text:   {"type":"motion","t":ms,"acc":[x,y,z],"gyro":[x,y,z]}
          {"type":"orientation","t":ms,"alpha":..,"beta":..,"gamma":..}
  acc is raw m/s^2 INCLUDING gravity - gravity is the absolute reference
  that levels the attitude estimate. gyro is deg/s on the wire, converted
  to rad/s here.

Two device conventions established by measurement, not documentation,
because iOS differs from both the spec and the usual descriptions:
  - accelerationIncludingGravity points DOWN in device axes, so world up
    is its negation.
  - rotationRate alpha/beta/gamma map onto device x/y/z in that order,
    not the spec's (z, x, y). See gyro_check.py, which verifies this
    against independent accelerometer and compass references.

Rotation tracking uses raw gyroscope, not the fused orientation angles:
the browser's alpha/beta/gamma hit gimbal lock at beta=+-90deg, which is
exactly where a phone held upright to film sits.
"""

import asyncio
import http
import json
import socket
import struct
import subprocess
import ssl
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import serve as ws_serve

CERT_DIR = Path(__file__).parent / "certs"
CERT_PATH = CERT_DIR / "cert.pem"
KEY_PATH = CERT_DIR / "key.pem"
WEB_PAGE_PATH = Path(__file__).parent / "web" / "index.html"

COMPASS_GAIN = 0.06          # fraction of yaw error corrected per reading
COMPASS_MAX_ACC_DEG = 25.0   # ignore readings less certain than this
COMPASS_MAX_TILT_DEG = 40.0  # heading is only well defined near-upright
ACC_TRUST_BAND = 0.8         # m/s^2 either side of g worth levelling from
GYRO_TRUST_RATE = 0.35       # rad/s where accelerometer trust halves
ACC_GAIN = 0.02              # levelling strength at full trust


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def get_tailscale_cert():
    """Returns (cert_path, key_path, hostname), or None to fall back to self-signed."""
    try:
        status = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        info = json.loads(status.stdout)
        hostname = info["Self"]["DNSName"].rstrip(".")
    except (FileNotFoundError, subprocess.TimeoutExpired,
            json.JSONDecodeError, KeyError):
        return None

    cert_path = CERT_DIR / "tailscale_cert.pem"
    key_path = CERT_DIR / "tailscale_key.pem"
    CERT_DIR.mkdir(exist_ok=True)
    print("Requesting a Tailscale certificate (can take several seconds)...")
    result = subprocess.run(
        ["tailscale", "cert",
         f"--cert-file={cert_path}", f"--key-file={key_path}", hostname],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"Tailscale certificate unavailable ({result.stderr.strip()}) "
              f"- falling back to a self-signed certificate.")
        return None

    return cert_path, key_path, hostname


def ensure_certificate(ip):
    CERT_DIR.mkdir(exist_ok=True)
    if CERT_PATH.exists():
        existing_sans = subprocess.run(
            ["openssl", "x509", "-in", str(CERT_PATH), "-noout", "-ext", "subjectAltName"],
            capture_output=True, text=True,
        ).stdout
        if f"IP Address:{ip}" in existing_sans:
            return

    print(f"Generating a new self-signed certificate for {ip} ...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(KEY_PATH), "-out", str(CERT_PATH),
        "-days", "3650", "-nodes",
        "-subj", "/CN=phone-stream",
        "-addext", f"subjectAltName=IP:{ip}",
    ], check=True, capture_output=True)


def euler_to_matrix(alpha_deg, beta_deg, gamma_deg):
    """W3C DeviceOrientation Z-X'-Y'' Euler angles -> rotation matrix (device -> world)."""
    a, b, g = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [ca * cg - sa * sb * sg, -cb * sa, ca * sg + sa * sb * cg],
        [sa * cg + ca * sb * sg, ca * cb, sa * sg - ca * sb * cg],
        [-cb * sg, sb, cb * cg],
    ])


class PhoneStream:
    # How far outside the gyro buffer an interval end may sit before the
    # rotation is refused rather than extrapolated.
    MAX_GYRO_GAP = 0.05   # seconds

    def __init__(self, https_port=8443, cert_http_port=8080, buffer_seconds=2.0):
        self.https_port = https_port
        self.cert_http_port = cert_http_port
        self._buffer_seconds = buffer_seconds
        self._lock = threading.Lock()

        self._latest_frame = None
        self._latest_frame_time = None
        self._latest_frame_capture = None
        self._frame_count = 0
        self._frame_arrival_times = deque(maxlen=30)
        self._accel = deque()   # (t, ax, ay, az) phone-local, m/s^2
        self._gyro = deque()    # (t, wx, wy, wz) phone-local, rad/s
        self._orient = deque()  # (t, R) world <- phone
        self._clock_offset = None

        # Attitude estimate: device frame -> world frame (world Z = up).
        # Gyro integrates it; the accelerometer's gravity direction levels
        # it. None until the first usable gravity reading initialises it.
        self._att_R = None
        self._gravity_dev = None   # low-passed raw accel = world up, in device axes
        self._gyro_bias = np.zeros(3)
        self._heading_offset = 0.0  # so heading reads 0 where the stream started

        # Compass: absolute yaw reference, the one thing gravity cannot
        # give. _compass_offset is our yaw minus the compass reading when
        # they were first compared; holding that difference steady is what
        # corrects yaw drift, and absorbs whatever fixed convention gap
        # exists between the two.
        self._compass = None        # (heading_deg, accuracy_deg)
        self._compass_offset = None
        self._compass_locked = False

        # Per-axis scalar integral of the gyro, for axis/sign diagnostics.
        # Integrated on arrival so no samples are missed.
        self._gyro_integral = np.zeros(3)

        self._html = WEB_PAGE_PATH.read_text()
        self._cert_path = None
        self._key_path = None

    def start(self):
        # Without this, prints can sit buffered when stdout isn't a terminal.
        sys.stdout.reconfigure(line_buffering=True)

        print("Setting up phone connection...")
        tailscale_cert = get_tailscale_cert()

        if tailscale_cert:
            self._cert_path, self._key_path, host = tailscale_cert
            print("Using a Tailscale-issued certificate - no manual trust step needed.")
            print(f"On your iPhone, visit  https://{host}:{self.https_port}/  "
                  f"and tap Start Streaming.")
        else:
            ip = get_local_ip()
            ensure_certificate(ip)
            self._cert_path, self._key_path = CERT_PATH, KEY_PATH

            cert_server = HTTPServer(("0.0.0.0", self.cert_http_port), self._make_cert_handler())
            threading.Thread(target=cert_server.serve_forever, daemon=True).start()

            print(f"1. On your iPhone, visit  http://{ip}:{self.cert_http_port}/  "
                  f"and install + trust the certificate (one-time).")
            print(f"2. Then visit  https://{ip}:{self.https_port}/  and tap Start Streaming.")

        threading.Thread(target=self._run_ws_server, daemon=True).start()

    def _make_cert_handler(self):
        class CertHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                cert_bytes = CERT_PATH.read_bytes()
                self.send_response(200)
                # Both headers needed for Safari to offer the install-profile flow.
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition",
                                  'attachment; filename="phone-stream.pem"')
                self.send_header("Content-Length", str(len(cert_bytes)))
                self.end_headers()
                self.wfile.write(cert_bytes)

        return CertHandler

    def _run_ws_server(self):
        asyncio.run(self._serve_forever())

    async def _serve_forever(self):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(str(self._cert_path), str(self._key_path))

        async with ws_serve(
            self._handle_connection,
            "0.0.0.0",
            self.https_port,
            ssl=ssl_context,
            process_request=self._serve_page,
        ):
            await asyncio.Future()

    async def _serve_page(self, connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        response = connection.respond(http.HTTPStatus.OK, self._html)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response

    async def _handle_connection(self, connection):
        async for message in connection:
            t_arrival = time.time()
            if isinstance(message, bytes):
                self._handle_frame(t_arrival, message)
            else:
                try:
                    self._handle_reading(t_arrival, json.loads(message))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

    def _handle_frame(self, t_arrival, message):
        capture_time_ms, = struct.unpack(">d", message[:8])
        jpeg_bytes = message[8:]

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        self._frame_count += 1
        self._frame_arrival_times.append(t_arrival)
        if self._frame_count % 30 == 0:
            latency_ms = t_arrival - capture_time_ms / 1000.0
            span = self._frame_arrival_times[-1] - self._frame_arrival_times[0]
            fps = (len(self._frame_arrival_times) - 1) / span if span > 0 else 0.0
            print(f"[phone_stream] frame {self._frame_count}: "
                  f"capture-to-arrival latency = {latency_ms * 1000:.0f}ms, "
                  f"delivery rate = {fps:.1f} fps")

        with self._lock:
            self._latest_frame = frame
            self._latest_frame_time = t_arrival
            # The phone stamps capture with the same Date.now() clock the
            # motion messages use, so the same offset maps it onto ours.
            # Needed to line a frame up against the gyro: arrival is late by
            # however long transit took, and over that gap the phone turns.
            self._latest_frame_capture = (
                capture_time_ms / 1000.0 + self._clock_offset
                if self._clock_offset is not None else t_arrival)

    def _handle_reading(self, t_arrival, reading):
        # Phone timestamps keep relative sample spacing accurate; an EMA'd
        # offset maps them onto our clock without per-message network jitter.
        phone_t = reading["t"] / 1000.0
        raw_offset = t_arrival - phone_t
        if self._clock_offset is None:
            self._clock_offset = raw_offset
        else:
            self._clock_offset = 0.9 * self._clock_offset + 0.1 * raw_offset
        t_local = phone_t + self._clock_offset

        with self._lock:
            if reading["type"] == "motion":
                ax, ay, az = reading["acc"]
                self._accel.append((t_local, ax, ay, az))
                self._trim(self._accel, t_local)
                if "gyro" in reading:
                    wx, wy, wz = np.deg2rad(reading["gyro"])
                    # Integrate on arrival, before appending, so no sample
                    # is missed regardless of how often callers poll. dt is
                    # bounded to skip startup and post-dropout gaps.
                    dt = t_local - self._gyro[-1][0] if self._gyro else 0.0
                    if 0.0 < dt < 0.5:
                        self._gyro_integral += np.array([wx, wy, wz]) * dt
                    self._gyro.append((t_local, wx, wy, wz))
                    self._trim(self._gyro, t_local)
                    self._update_attitude(np.array([ax, ay, az]),
                                          np.array([wx, wy, wz]),
                                          dt if 0.0 < dt < 0.5 else 0.0)
            elif reading["type"] == "orientation":
                R = euler_to_matrix(reading["alpha"], reading["beta"], reading["gamma"])
                self._orient.append((t_local, R))
                self._trim(self._orient, t_local)
                if reading.get("compass") is not None:
                    self._compass = (float(reading["compass"]),
                                     reading.get("compassAcc"))
                    self._apply_compass(*self._compass)

    def _trim(self, buf, t_now):
        while buf and t_now - buf[0][0] > self._buffer_seconds:
            buf.popleft()

    def get_latest_frame(self):
        """Returns (frame, arrival_time, capture_time). All may be None."""
        with self._lock:
            return (self._latest_frame, self._latest_frame_time,
                    self._latest_frame_capture)

    def get_rotation(self, t0, t1):
        """
        Gyro-only rotation over [t0, t1]: (R_delta, mean_rate_rad_s, ok).

        R_delta maps a device-frame vector at t1 into the device frame at
        t0, matching get_delta(). Bias-corrected but NOT levelled by
        gravity - over a single frame interval the gyro alone is far more
        accurate than the fused attitude, which is deliberately slow.

        Integrates over EXACTLY [t0, t1]. Summing only the samples that
        fall inside the interval silently drops the gap between t0 and the
        first sample, and between the last and t1 - at 60Hz gyro against
        15fps frames that is about a quarter of the interval, which during
        a 60 deg/s pan leaves ~1 deg of rotation un-removed. One degree is
        ~11px of image shift, which the flow fit then reports as sideways
        translation. So the rate is interpolated to the interval ends and
        the ends are integrated with the rest.
        """
        if t1 <= t0:
            return np.eye(3), 0.0, False

        with self._lock:
            samples = list(self._gyro)
            bias = self._gyro_bias.copy()

        if len(samples) < 2:
            return np.eye(3), 0.0, False

        ts = np.array([s[0] for s in samples])
        w = np.array([s[1:] for s in samples]) - bias

        # np.interp holds the end value beyond the data, so a short gap
        # either side coasts on the nearest rate. Past that the rotation
        # was never measured and guessing it would be worse than refusing.
        if max(ts[0] - t0, t1 - ts[-1], 0.0) > self.MAX_GYRO_GAP:
            return np.eye(3), 0.0, False

        # Knots: both interval ends, plus every sample strictly between.
        knots = np.concatenate(([t0], ts[(ts > t0) & (ts < t1)], [t1]))
        wk = np.column_stack([np.interp(knots, ts, w[:, i]) for i in range(3)])

        R = np.eye(3)
        for i in range(1, len(knots)):
            dt = knots[i] - knots[i - 1]
            if dt <= 0.0:
                continue
            # Midpoint rate rather than the left-hand one: with only a few
            # samples per frame, forward Euler lags a changing rate badly.
            R = R @ cv2.Rodrigues(0.5 * (wk[i - 1] + wk[i]) * dt)[0]

        return R, float(np.linalg.norm(wk, axis=1).mean()), True

    def get_latest_imu(self):
        """
        Most recent raw sample: ((ax, ay, az), (wx, wy, wz), rate_hz), or
        None if nothing is buffered. Accel m/s^2, gyro rad/s. Rate is
        measured over the buffer, so it reflects what actually arrived.
        """
        with self._lock:
            if not self._accel or not self._gyro:
                return None
            _, ax, ay, az = self._accel[-1]
            _, wx, wy, wz = self._gyro[-1]
            span = self._accel[-1][0] - self._accel[0][0]
            rate = (len(self._accel) - 1) / span if span > 0 else 0.0
        return (ax, ay, az), (wx, wy, wz), rate

    def get_orientation(self):
        """
        Returns (heading_deg, tilt_deg, pitch_deg, ready).

        heading_deg: facing direction clockwise from where the stream
          started, measured about true vertical. Relative, not compass
          absolute - no magnetometer, so there is no north reference.
        tilt_deg: how far from upright, unsigned, 0..180.
        pitch_deg: signed version of the same tip, -180..180: 0 upright,
          +90 flat with the camera down, -90 flat with the camera up,
          180 inverted. tilt cannot tell those two 90s apart; this can.
        ready: False until gravity has initialised the attitude estimate.
        """
        with self._lock:
            if self._att_R is None:
                return 0.0, 0.0, 0.0, False
            R = self._att_R.copy()
            offset = self._heading_offset

        heading = (self._heading_from(R) - offset) % 360.0
        # Device Y points up the screen; compare it to world up.
        up_world = R @ np.array([0.0, 1.0, 0.0])
        tilt = float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0))))
        # World up seen from the device: its Y/Z split gives the sign.
        up_dev = R.T @ np.array([0.0, 0.0, 1.0])
        pitch = float(np.degrees(np.arctan2(up_dev[2], up_dev[1])))
        return float(heading), tilt, pitch, True

    @staticmethod
    def _heading_from(R):
        """Yaw of the rear camera (device -Z) about world up, in degrees."""
        forward = R @ np.array([0.0, 0.0, -1.0])
        return float(np.degrees(np.arctan2(forward[0], forward[1])))

    def _update_attitude(self, accel, gyro, dt):
        """
        Complementary filter. The gyro carries the estimate between
        samples; the accelerometer's gravity direction pulls the tilt back
        toward truth so it cannot drift away. Caller holds the lock.

        Gravity fixes roll and pitch absolutely, but says nothing about
        rotation *around* vertical - so heading drift is reduced (tilt
        error no longer leaks into it, and gyro bias is subtracted) but
        not eliminated. Only a magnetometer or vision can fully pin yaw.
        """
        a_norm = float(np.linalg.norm(accel))
        if a_norm < 1e-6:
            return

        # Low-pass the raw accel to track gravity through ordinary motion.
        if self._gravity_dev is None:
            self._gravity_dev = accel.copy()
        else:
            self._gravity_dev = 0.95 * self._gravity_dev + 0.05 * accel

        # On iOS, accelerationIncludingGravity points DOWN in device axes,
        # so world up is its negation. Established from the device, not
        # documentation: held upright the tilt angle read 180 degrees
        # instead of 0, which is precisely a flipped up-axis. (A flipped
        # sign is invisible when the phone is flat - both give 90 - so it
        # only shows at the upright and inverted extremes.)
        #
        # Correction uses the RAW direction, not the low-passed one: the
        # small gain below already does the filtering, and pre-filtering
        # only adds lag that fights correct gyro integration during
        # sustained rotation.
        up_meas = -accel / a_norm

        if self._att_R is None:
            self._att_R = self._level_from(
                -self._gravity_dev / np.linalg.norm(self._gravity_dev))
            self._heading_offset = self._heading_from(self._att_R)
            return

        # Still and level? Then the gyro should read zero, so whatever it
        # does read is bias. Tracking and subtracting it is the main lever
        # on heading drift.
        stationary = abs(a_norm - 9.81) < 0.35 and float(np.linalg.norm(gyro)) < 0.05
        if stationary:
            self._gyro_bias = 0.98 * self._gyro_bias + 0.02 * gyro

        if dt > 0.0:
            step = cv2.Rodrigues((gyro - self._gyro_bias) * dt)[0]
            self._att_R = self._att_R @ step

        # Only trust the accelerometer as a gravity reference when it is
        # plausibly measuring just gravity. Two ways it is not:
        #
        #  - magnitude far from g: the phone is being accelerated.
        #  - the phone is rotating: swinging it through a yaw arc adds
        #    centripetal acceleration pointing horizontally, which tilts
        #    the apparent "down" and shows up as phantom pitch that
        #    settles only once you stop. Note the magnitude test alone
        #    does NOT catch this - centripetal adds vectorially, so 2m/s2
        #    sideways only moves |a| from 9.81 to 10.0.
        #
        # While rotating we simply coast on the gyro, which is what it is
        # good at, and resume levelling when things settle.
        # Grade how far the accelerometer is trusted rather than switching
        # it off. It is only measuring pure gravity when the phone is
        # neither accelerating nor swinging through an arc, so trust
        # decays with both - but never reaches zero, so there is always
        # some restoring force and pitch keeps tracking through a turn
        # instead of freezing. With the gyro axes correct it carries the
        # motion accurately in the meantime.
        rate = float(np.linalg.norm(gyro - self._gyro_bias))
        trust = (1.0 / (1.0 + (rate / GYRO_TRUST_RATE) ** 2)
                 / (1.0 + (abs(a_norm - 9.81) / ACC_TRUST_BAND) ** 2))

        if trust > 0.01:
            up_est = self._att_R.T @ np.array([0.0, 0.0, 1.0])
            correction = np.cross(up_meas, up_est) * ACC_GAIN * trust
            self._att_R = self._att_R @ cv2.Rodrigues(correction)[0]

    def _apply_compass(self, compass_deg, accuracy):
        """
        Steer yaw so the gap between our heading and the compass stays at
        whatever it was when they first met. Correcting the *gap* rather
        than the absolute value means any fixed convention difference
        cancels out - it only needs that gap to be stable. Caller holds
        the lock.

        Gated hard, because a bad compass reading is worse than none:
        magnetometers are thrown off by nearby metal and electronics, and
        the heading is only well defined near-upright.
        """
        if self._att_R is None:
            return
        if accuracy is None or accuracy < 0 or accuracy > COMPASS_MAX_ACC_DEG:
            self._compass_locked = False
            return

        up_world = self._att_R @ np.array([0.0, 1.0, 0.0])
        tilt = np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0)))
        if tilt > COMPASS_MAX_TILT_DEG:
            self._compass_locked = False
            return

        ours = self._heading_from(self._att_R)
        gap = (compass_deg - ours + 180.0) % 360.0 - 180.0

        if self._compass_offset is None:
            self._compass_offset = gap
            self._compass_locked = True
            return

        # Positive err means we have drifted anticlockwise of the compass.
        err = (gap - self._compass_offset + 180.0) % 360.0 - 180.0
        # Rotating the attitude about world +Z by theta *decreases* our
        # heading, so the correction takes the opposite sign.
        theta = -np.radians(err) * COMPASS_GAIN
        self._att_R = cv2.Rodrigues(np.array([0.0, 0.0, theta]))[0] @ self._att_R
        self._compass_locked = True

    def get_attitude_matrix(self):
        """
        Full device->world rotation (world Z up), or None before gravity
        initialises it. Roll and pitch come from gravity, yaw from the
        compass - both absolute - with the gyro carrying it smoothly
        between readings.
        """
        with self._lock:
            return None if self._att_R is None else self._att_R.copy()

    def get_compass(self):
        """Returns (heading_deg, accuracy_deg, correcting) or None."""
        with self._lock:
            if self._compass is None:
                return None
            heading, accuracy = self._compass
            return heading, accuracy, self._compass_locked

    @staticmethod
    def _level_from(up_dev):
        """
        Build an initial device->world rotation from the measured up
        direction alone. Yaw is unobservable from gravity, so it comes out
        arbitrary - get_orientation() subtracts it off as the zero point.
        """
        world_up = np.array([0.0, 0.0, 1.0])
        axis = np.cross(up_dev, world_up)
        s = float(np.linalg.norm(axis))
        c = float(np.dot(up_dev, world_up))
        if s < 1e-8:
            return np.eye(3) if c > 0 else cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
        return cv2.Rodrigues(axis / s * np.arctan2(s, c))[0]

    def get_delta(self, t0, t1):
        """
        Returns (R_delta, displacement_m, have_data) for the interval [t0, t1].
        R_delta maps a vector from the phone frame at t1 into the phone frame
        at t0 (same convention as cv2.recoverPose).
        """
        with self._lock:
            accel_samples = [s for s in self._accel if t0 <= s[0] <= t1]
            gyro_samples = [s for s in self._gyro if t0 <= s[0] <= t1]
            orient_samples = [s for s in self._orient if t0 <= s[0] <= t1]
            gravity = None if self._gravity_dev is None else self._gravity_dev.copy()

        if len(accel_samples) < 2 or len(gyro_samples) < 2 or len(orient_samples) < 2:
            return np.eye(3), np.zeros(3), False

        # Exponential map: compose per-sample Rodrigues(omega * dt) in order.
        R_delta = np.eye(3)
        for i in range(1, len(gyro_samples)):
            t_prev, wx, wy, wz = gyro_samples[i - 1]
            t_curr = gyro_samples[i][0]
            dt = t_curr - t_prev
            R_delta = R_delta @ cv2.Rodrigues(np.array([wx, wy, wz]) * dt)[0]

        # Rotate accel into world frame, then double-integrate (trapezoid),
        # assuming zero velocity at window start. accel arrives raw, so
        # gravity has to come off first or it dominates the integral.
        R_start = orient_samples[0][1]
        R_mid = orient_samples[len(orient_samples) // 2][1]
        t_accel = np.array([s[0] for s in accel_samples])
        a_local = np.array([[s[1], s[2], s[3]] for s in accel_samples])
        if gravity is not None:
            a_local = a_local - gravity
        a_world = a_local @ R_mid.T

        dt = np.diff(t_accel)
        a_avg = 0.5 * (a_world[1:] + a_world[:-1])
        v = np.vstack([np.zeros(3), np.cumsum(a_avg * dt[:, None], axis=0)])
        v_avg = 0.5 * (v[1:] + v[:-1])
        displacement_world = np.sum(v_avg * dt[:, None], axis=0)

        displacement = R_start.T @ displacement_world
        return R_delta, displacement, True


if __name__ == "__main__":
    stream = PhoneStream()
    stream.start()
    print("Running standalone (Ctrl+C to stop).")
    try:
        while True:
            frame, _, _ = stream.get_latest_frame()
            if frame is not None:
                cv2.imshow("phone_stream.py standalone preview", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
