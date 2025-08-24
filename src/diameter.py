import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# --- Define the pixel-to-millimeter conversion ratio ---
PIXEL_TO_MM_RATIO = 0.1

def get_diameter(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=150
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            diameter_pixels = i[2] * 2
            diameter_mm = diameter_pixels * PIXEL_TO_MM_RATIO
            diameter_cm = diameter_mm / 10
            
            cv2.circle(undistorted_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(undistorted_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            cv2.putText(undistorted_frame, f"Diameter: {diameter_cm:.2f} cm", (i[0] + 20, i[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return undistorted_frame, diameter_cm

    return undistorted_frame, None

if __name__ == '__main__':
    # --- Upload the calibration data ---
    print("Please upload the 'camera_calibration.npz' file.")
    uploaded_calib = files.upload()

    calib_file_name = next((f for f in uploaded_calib if f.endswith('.npz')), None)
    
    if calib_file_name:
        try:
            calib_data = np.load(calib_file_name)
            camera_matrix = calib_data['mtx']
            dist_coeffs = calib_data['dist']
            print("Calibration data loaded successfully.")
            
            # --- Upload the image ---
            print("\nPlease upload the image you want to test.")
            uploaded_image = files.upload()

            if uploaded_image:
                image_path = next(iter(uploaded_image))
                print(f"Successfully uploaded: '{image_path}'")
                
                try:
                    distorted_image = cv2.imread(image_path)
                    if distorted_image is None:
                        raise FileNotFoundError
                    
                    result_image, measured_diameter_cm = get_diameter(distorted_image, camera_matrix, dist_coeffs)
                    
                    if measured_diameter_cm is not None:
                        print(f"Measured Diameter: {measured_diameter_cm:.2f} cm")
                    else:
                        print("No disc found in the image.")
                    
                    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                    plt.title('Diameter Measurement')
                    plt.axis('off')
                    plt.show()

                    os.remove(image_path)
                    os.remove(calib_file_name)
                    
                except FileNotFoundError:
                    print(f"Error: Image file not found at '{image_path}'")
            else:
                print("No image file was uploaded.")

        except KeyError:
            print("Error: The .npz file does not contain the expected keys 'mtx' or 'dist'.")
        except Exception as e:
            print(f"An error occurred while loading the calibration file: {e}")
            
    else:
        print("No .npz file was uploaded. Please try again.")
