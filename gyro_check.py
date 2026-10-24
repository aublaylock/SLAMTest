"""
Gyro axis and sign verification.

Walks through three deliberate motions and compares the gyro's integrated
rotation against an independent reference for each: the accelerometer for
the two tilting motions, the compass for the turn. That pins down which
gyro component maps to which device axis, its sign, and its scale.

Move SLOWLY - the references assume the phone is not being accelerated,
so a gentle motion keeps centripetal error out of the comparison.

    ./venv/bin/python gyro_check.py
"""

import numpy as np
import cv2

from phone_stream import PhoneStream

FONT = cv2.FONT_HERSHEY_SIMPLEX
W, H = 760, 420

STEPS = [
    ("PITCH", "Hold upright, then tip the TOP AWAY from you",
     "until flat, screen up / camera down (~90 deg)", "accel pitch"),
    ("YAW", "Hold upright, then turn 90 deg TO YOUR RIGHT",
     "stay upright, keep the phone close to your body", "compass"),
    ("ROLL", "Hold upright, then roll so the RIGHT EDGE goes DOWN",
     "about 90 deg, screen still facing you", "accel roll"),
]


def accel_angles(accel):
    """(pitch, roll) in degrees from the gravity direction."""
    ax, ay, az = accel
    return (float(np.degrees(np.arctan2(-az, -ay))),
            float(np.degrees(np.arctan2(-ax, -ay))))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def reference_value(stream, kind):
    imu = stream.get_latest_imu()
    if imu is None:
        return None
    pitch, roll = accel_angles(imu[0])
    if kind == "accel pitch":
        return pitch
    if kind == "accel roll":
        return roll
    compass = stream.get_compass()
    return None if compass is None else compass[0]


def main():
    stream = PhoneStream()
    stream.start()
    print("Waiting for the phone... open the https:// URL and tap Start Streaming.")
    print("Then follow the on-screen steps. SPACE to start/stop each, q to quit.\n")

    step = 0
    recording = False
    ref_start = None
    results = []

    while True:
        canvas = np.zeros((H, W, 3), np.uint8)
        imu = stream.get_latest_imu()

        if imu is None:
            cv2.putText(canvas, "waiting for phone...", (20, 60), FONT, 0.9,
                        (0, 140, 255), 2)
        elif step >= len(STEPS):
            cv2.putText(canvas, "All steps done - results in the terminal.",
                        (20, 60), FONT, 0.8, (120, 255, 120), 2)
            cv2.putText(canvas, "Press q to quit.", (20, 100), FONT, 0.7,
                        (180, 180, 180), 1)
        else:
            name, line1, line2, ref_kind = STEPS[step]
            (ax, ay, az), (wx, wy, wz), rate = imu
            gx, gy, gz = np.degrees([wx, wy, wz])
            integral = stream.get_gyro_integral()
            pitch, roll = accel_angles((ax, ay, az))

            cv2.putText(canvas, f"STEP {step + 1}/{len(STEPS)}:  {name}", (20, 44),
                        FONT, 0.95, (255, 210, 120), 2)
            cv2.putText(canvas, line1, (20, 80), FONT, 0.6, (220, 220, 220), 1)
            cv2.putText(canvas, line2, (20, 106), FONT, 0.6, (220, 220, 220), 1)

            colour = (120, 255, 120) if recording else (150, 150, 150)
            cv2.putText(canvas,
                        "RECORDING - do the motion, then SPACE" if recording
                        else "SPACE to start recording this step",
                        (20, 146), FONT, 0.7, colour, 2)

            cv2.putText(canvas, "gyro rate (deg/s)", (20, 196), FONT, 0.6,
                        (150, 150, 150), 1)
            for i, (lbl, v) in enumerate((("x", gx), ("y", gy), ("z", gz))):
                # Highlight whichever axis is currently dominant.
                hot = abs(v) > 12 and abs(v) >= max(abs(gx), abs(gy), abs(gz))
                cv2.putText(canvas, f"{lbl} {v:+8.1f}", (20 + i * 150, 232), FONT,
                            0.8, (120, 255, 120) if hot else (200, 200, 200), 2)

            cv2.putText(canvas, "integrated since reset (deg)", (20, 282), FONT,
                        0.6, (150, 150, 150), 1)
            for i, (lbl, v) in enumerate(zip("xyz", integral)):
                big = abs(v) >= max(abs(integral)) and abs(v) > 15
                cv2.putText(canvas, f"{lbl} {v:+8.1f}", (20 + i * 150, 318), FONT,
                            0.8, (255, 210, 120) if big else (200, 200, 200), 2)

            ref = reference_value(stream, ref_kind)
            ref_txt = "n/a" if ref is None else f"{ref:+.1f}"
            cv2.putText(canvas,
                        f"reference ({ref_kind}): {ref_txt}    "
                        f"accel pitch {pitch:+.0f}  roll {roll:+.0f}    IMU {rate:.0f}Hz",
                        (20, 370), FONT, 0.55, (180, 180, 180), 1)

        cv2.imshow("Gyro axis check", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" ") and imu is not None and step < len(STEPS):
            if not recording:
                stream.reset_gyro_integral()
                ref_start = reference_value(stream, STEPS[step][3])
                recording = True
            else:
                integral = stream.get_gyro_integral()
                ref_end = reference_value(stream, STEPS[step][3])
                delta = (None if ref_start is None or ref_end is None
                         else wrap(ref_end - ref_start))
                results.append((STEPS[step][0], STEPS[step][3], integral, delta))
                recording = False
                step += 1

    cv2.destroyAllWindows()

    if results:
        print("\n" + "=" * 62)
        print("REPORT THIS OUTPUT")
        print("=" * 62)
        for name, ref_kind, integral, delta in results:
            d = "n/a" if delta is None else f"{delta:+.1f}"
            print(f"\n{name}")
            print(f"  gyro integral  x={integral[0]:+8.1f}  "
                  f"y={integral[1]:+8.1f}  z={integral[2]:+8.1f}   deg")
            print(f"  {ref_kind} change: {d} deg")
        print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
