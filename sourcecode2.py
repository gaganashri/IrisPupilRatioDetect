import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define eye and iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Open webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 's' to capture a frame and calculate the iris-pupil ratio.")
print("Press 'q' to quit.")

selected_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    frame = cv.flip(frame, 1)
    cv.putText(frame, "Press 's' to capture frame, 'q' to quit.", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.imshow("Video Feed", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        selected_frame = frame.copy()
        print("Frame captured. Processing...")
        break
    elif key == ord('q'):
        print("Exiting...")
        cap.release()
        cv.destroyAllWindows()
        exit()

cap.release()
cv.destroyAllWindows()

if selected_frame is not None:
    gray_frame = cv.cvtColor(selected_frame, cv.COLOR_BGR2GRAY)
    results = face_mesh.process(cv.cvtColor(selected_frame, cv.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        img_h, img_w = selected_frame.shape[:2]
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                                for p in results.multi_face_landmarks[0].landmark])

        ipd_pixels = np.linalg.norm(mesh_points[LEFT_EYE_OUTER] - mesh_points[RIGHT_EYE_OUTER])
        ipd_mm = 63  # Average adult IPD
        pixel_to_mm = ipd_mm / ipd_pixels

        def process_iris(iris_landmarks, eye_side):
            global selected_frame

            try:
                (cx, cy), iris_radius_px = cv.minEnclosingCircle(mesh_points[iris_landmarks])
            except IndexError:
                print(f"Failed to locate iris landmarks for {eye_side}.")
                return

            iris_center = np.array([cx, cy], dtype=np.int32)
            iris_radius_mm = iris_radius_px * pixel_to_mm

            x1, y1 = max(0, iris_center[0] - int(iris_radius_px)), max(0, iris_center[1] - int(iris_radius_px))
            x2, y2 = min(img_w, iris_center[0] + int(iris_radius_px)), min(img_h, iris_center[1] + int(iris_radius_px))

            iris_roi = gray_frame[y1:y2, x1:x2]

            if iris_roi.size != 0:
                iris_gray = cv.equalizeHist(iris_roi)
                blurred = cv.GaussianBlur(iris_gray, (5, 5), 0)
                thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv.THRESH_BINARY_INV, 11, 2)
                edges = cv.Canny(blurred, 50, 150)
                combined = cv.bitwise_and(thresh, edges)
                contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                pupil_radius_mm = 0
                pupil_center = (int(iris_center[0]), int(iris_center[1]))

                if contours:
                    largest_area = 0
                    best_ellipse = None
                    for contour in contours:
                        if len(contour) >= 5:
                            ellipse = cv.fitEllipse(contour)
                            (x, y), (MA, ma), angle = ellipse
                            area = np.pi * (MA / 2) * (ma / 2)
                            if area > largest_area:
                                largest_area = area
                                best_ellipse = ellipse

                    if best_ellipse:
                        (x, y), (MA, ma), angle = best_ellipse
                        radius_px = (MA + ma) / 4
                        min_pupil_radius_px = iris_radius_px * 0.2
                        max_pupil_radius_px = iris_radius_px * 0.7

                        if min_pupil_radius_px <= radius_px <= max_pupil_radius_px:
                            pupil_radius_mm = radius_px * pixel_to_mm
                            pupil_center = (int(x), int(y))
                        else:
                            pupil_radius_mm = iris_radius_mm / 3
                    else:
                        pupil_radius_mm = iris_radius_mm / 3

                iris_pupil_ratio = iris_radius_mm / pupil_radius_mm if pupil_radius_mm > 0 else 0
                iris_pupil_ratio = max(2.0, min(iris_pupil_ratio, 5.0))

                print(f"{eye_side} Iris Radius (mm): {iris_radius_mm:.2f}")
                print(f"{eye_side} Pupil Radius (mm): {pupil_radius_mm:.2f}")
                print(f"{eye_side} Iris-to-Pupil Ratio: {iris_pupil_ratio:.2f}")

                annotated_frame = selected_frame.copy()
                cv.circle(annotated_frame, tuple(iris_center), int(iris_radius_px), (0, 0, 255), 2)
                cv.circle(annotated_frame, tuple(pupil_center), int(pupil_radius_mm / pixel_to_mm), (0, 255, 0), 2)
                cv.imwrite(f'{eye_side.lower()}_annotated_iris.png', annotated_frame)
                print(f"Saved annotation for {eye_side}.")

        process_iris(LEFT_IRIS, "Left Eye")
        process_iris(RIGHT_IRIS, "Right Eye")
    else:
        print("No face landmarks detected.")
else:
    print("No frame was selected for processing.")
