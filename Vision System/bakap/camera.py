from camera_utils import (
    create_pipeline,
    find_red_blob_centers,
    draw_centers,
    save_centers_to_csv,
    on_mouse_move,
    draw_hsv_at_cursor,
    find_green_blob_center,
    get_intrinsics,
    draw_distances
)
import numpy as np
import cv2


def main():
    pipeline = create_pipeline()
    intrinsics = get_intrinsics(pipeline)
    print("Controls:  S = save points   |   Q = quit")

    cv2.namedWindow("Red Blob Detector")
    cv2.setMouseCallback("Red Blob Detector", on_mouse_move)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            green_center = find_green_blob_center(frame)
            red_centers = find_red_blob_centers(frame)

            # draw_centers only draws dots now (no pixel labels)
            # frame = draw_centers(frame, red_centers)

            if green_center:
                cv2.circle(frame, green_center, 8, (0, 255, 0), -1)
                cv2.putText(frame, "BASE", (green_center[0] + 8, green_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw mm distances next to each red blob (replaces pixel label)
            frame = draw_distances(frame, green_center, red_centers, intrinsics, Z=85)

            frame = draw_hsv_at_cursor(frame)
            cv2.imshow("Red Blob Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_centers_to_csv(red_centers)
            elif key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()