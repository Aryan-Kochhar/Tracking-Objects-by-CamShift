import numpy as np  # for numerical processing
import cv2  # binding with OpenCV library
import argparse  # for parsing command-line arguments

# Initializing the frames of the video with ROI points, and input mode
frame = None
roiPts = []  # Stores the selected ROI points
inputMode = False  # Determines if the user is selecting ROI points
camera = None  # Video source (camera or video file)

# Function to handle mouse clicks for selecting ROI points
def selectROI(event, x, y, flags, param):
    global frame, roiPts, inputMode
    # If in ROI selection mode, check for left-click and ensure only 4 points are selected
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))  # Store the clicked point
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)  # Draw a small circle at the selected point
        cv2.imshow("frame", frame)  # Update the display

def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", help="Path to (optional) video file")
    args = vars(arg_parse.parse_args())

    global frame, roiPts, inputMode, camera

    # If no video file is provided, grab reference to the camera
    if args.get("video", None) is None:
        camera = cv2.VideoCapture(0)  # Open webcam
    else:
        camera = cv2.VideoCapture(args["video"])  # Open video file

    # Set up a normal resizable window
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  
    cv2.setMouseCallback("frame", selectROI)  # Set up mouse callback for selecting ROI

    # Taking 10 iterations or movement of at least 1 pixel along the box of ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None  # Bounding box for object tracking

    while True:
        grabbed, frame = camera.read()  # Capture a frame from the camera/video
        if not grabbed:  # If the video has ended or the camera is not accessible
            break

        # If ROI has been computed, apply CamShift for tracking
        if roiBox is not None:
            # Convert the current frame to HSV color space and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)  # Create back projection

            # Apply CamShift to the back projection, convert the points to a bounding box, and draw it
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))  # Convert tracking box points
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)  # Draw bounding box on the tracked object

        # Show the frame
        cv2.imshow("frame", frame)

        # Press 's' to select ROI
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(roiPts) < 4:
            inputMode = True  # Enable ROI selection mode
            print("[INFO] Select 4 points for ROI...")  # Debugging message

            # Wait for the user to select 4 points
            while len(roiPts) < 4:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            inputMode = False  # Disable ROI selection mode

            # Compute the bounding box from the selected ROI points
            roiPts_np = np.array(roiPts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(roiPts_np)  # Get bounding box coordinates
            roiBox = (x, y, w, h)  # Save it for tracking

            # Convert the selected region to HSV and compute the histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            roi = hsv[y:y+h, x:x+w]  # Extract the selected ROI
            roiHist = cv2.calcHist([roi], [0], None, [180], [0, 180])  # Compute histogram
            cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)  # Normalize the histogram
            print("[INFO] ROI Selected. Tracking started...")  # Debugging message

        # Press 'q' to exit
        elif key == ord("q"):
            break

    # Cleanup: Release the camera/video and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
