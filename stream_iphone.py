"""
Minimal viewer: shows the phone's camera feed with no SLAM processing.
Useful for confirming the phone/laptop connection works. See
phone_stream.py for the one-time certificate setup.

    ./venv/bin/python stream_iphone.py
"""

import time
import cv2

from phone_stream import PhoneStream


def main():
    stream = PhoneStream()
    stream.start()

    print("Waiting for the phone to connect... Press 'q' in the window to quit.")

    last_seen_frame_time = None
    while True:
        frame, frame_time, _ = stream.get_latest_frame()
        if frame is None or frame_time == last_seen_frame_time:
            time.sleep(0.005)
            continue
        last_seen_frame_time = frame_time

        cv2.imshow("iPhone Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
