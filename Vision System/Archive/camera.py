from camera_utils import (
    create_pipeline,
    find_red_blob_centers,
    draw_centers,
    save_centers_to_csv,
    on_mouse_move,
    draw_hsv_at_cursor,
    find_green_blob_center,
    get_intrinsics,
    draw_distances,
    fit_two_lines_ransac,   # ← new
    draw_ransac_lines,      # ← new
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

            # green_center = find_green_blob_center(frame)
            red_centers  = find_red_blob_centers(frame)

            # if green_center:
            #     cv2.circle(frame, green_center, 8, (0, 255, 0), -1)
            #     cv2.putText(frame, "BASE", (green_center[0] + 8, green_center[1]),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Fit two lines with RANSAC and draw them
            # epsilon = inlier threshold in pixels — increase if points are noisier
            if len(red_centers) >= 4:
                lines = fit_two_lines_ransac(red_centers, epsilon=15, iterations=200)
                frame = draw_ransac_lines(frame, lines)
            else:
                # Not enough points yet — just draw raw dots
                frame = draw_centers(frame, red_centers)

            # frame = draw_distances(frame, green_center, red_centers, intrinsics, Z=85)
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
