# HEADER ###############################################
# ANDREW PIERSON's TENNIS BALL SPEED REGRESSION MODEL
# FILENAME: BallTracker.py
# DATE: 4/23/2024
# DESCRIPTION: This python script uses OpenCV to identify a tennis ball
#               and then uses linear regression to predict the speed of
#               the ball using self-collected displacement and speed data.
# INSTRUCTIONS ##########################################
# 1. Run the following command LINE in terminal or run customized window.
# 2. Or simply just run the code to do live camera detection
# Command LINE:    python Balltracker.py --video VideoTest.mp4
# Run customized:  --video VideoTest.mp4
# Sources ###############################################
# Source 1- Jonathan Foster's code using OpenCV Ball_tracking.py code to see the ball.
#           Includes color_picker.py to test HSV values.
#   link: https://github.com/jonathanfoster/ball-tracking/blob/master/README.md
# Source 2- Famous Professor, Mark Terwilliger's linear regression code.
#   link: https://una.instructure.com/courses/86685/files/14886587?wrap=1
# Source 3- Stack Overflow- zoom into frame code
#   link: https://stackoverflow.com/questions/50870405/how-can-i-zoom-my-webcam-in-open-cv-python
#
# Libraries #############################################
import numpy as np  # for reading training data
from sklearn import linear_model  # metrics for linear regression
import sklearn.metrics as sm  # metrics for linear regression
import argparse  # parsing video argument
import imutils  # resizing the frame (zoom)
import cv2  # for computer vision
import matplotlib.pyplot as plt  # for plotting training data
import tkinter as tk  # for GUI

class BallTracker:
    def __init__(self):
        # Values to change HSV and Camera Zoom
        self.scale = 25  # Camera Zoom
        self.greenLower = (35, 33, 184)  # Upper Bound of HSV
        self.greenUpper = (96, 141, 250)  # Lower Bound of HSV
        # values to make the calculations work
        self.framecounter = 0
        self.frame_count = 2
        self.totalframes = 0
        self.times = []
        self.totalsec = 0
        self.dispXY = []
        self.speeds = []
        self.speedtotal = 0
        self.Maxspeed = 0
        self.averagespeed = 0
        self.prevX = None
        self.prevY = None

        # Parse arguments
        self.args = self.parse_arguments()
        # Setup camera
        self.camera = self.setup_camera()
        # Setup GUI
        self.main_window, self.speed_label = self.setup_gui()
        # Setup fps
        self.fps = self.setup_fps()
        # Load training data
        (self.X_train, self.y_train, self.X_test, self.y_test,
         self.linear_regressor, self.y_test_pred) = self.load_training_data()
        
    def parse_arguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the (optional) video file")
        args = vars(ap.parse_args())
        return args
    
    def setup_camera(self):
        if not self.args.get("video", False):
            camera = cv2.VideoCapture(0)
            print("**no video argument**")
        else:
            camera = cv2.VideoCapture(self.args["video"])
        return camera

    def setup_gui(self):
        main_window = tk.Tk()
        main_window.geometry("750x750")
        main_window.title("Speed Detection")
        speed_label = tk.Label(main_window, text="Speed: 0", font=('Helvetica', 82))
        speed_label.pack()
        return main_window, speed_label

    def setup_fps(self):
        fpsI = self.camera.get(cv2.CAP_PROP_FPS)
        if 25 < fpsI < 35:
            fps = 30
        elif 55 < fpsI < 65:
            fps = 60
        else:
            fps = 30  # Default FPS if unable to get from camera
        print(f"Frames per second: {fps}")
        return fps

    def load_training_data(self):
        input_file = 'tennisdata.txt'
        data = np.loadtxt(input_file, delimiter=',', usecols=(0, 1))
        X, y = data[:, :-1], data[:, -1]
        num_training = int(0.7 * len(X))
        num_test = len(X) - num_training
        X_train, y_train = X[:num_training], y[:num_training]
        X_test, y_test = X[num_training:], y[num_training:]
        linear_regressor = linear_model.LinearRegression()
        linear_regressor.fit(X_train, y_train)
        y_test_pred = linear_regressor.predict(X_test)
        plt.scatter(X_test, y_test, color='red', label='Test data')
        plt.plot(X_test, y_test_pred, color='black', linewidth=4)
        plt.show()
        return X_train, y_train, X_test, y_test, linear_regressor, y_test_pred

    def process_frames(self):
        while True:
            # Grab video frame - Source 1
            (grabbed, frame) = self.camera.read()
            if self.args.get("video") and not grabbed:
                break
            self.totalframes += 1

            # Calculate time in seconds
            if self.totalframes == self.fps:
                self.totalframes = 0
                self.totalsec += 1

            # Process the frame
            self.process_frame(frame)

            # Break if 'q' is pressed - Source 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Release camera and windows - Source 1
        self.camera.release()
        cv2.destroyAllWindows()

        # Update GUI with average speed
        self.update_gui_final()

        # Write results
        self.write_results()

        # Compute performance metrics
        self.compute_performance_metrics()

    def process_frame(self, frame):
        # Resize frame by scaling (zooming in) - Sources 1, 3
        frame = imutils.resize(frame, width=600)
        height, width = frame.shape[:2]
        centerX, centerY = int(width / 2), int(height / 2)
        radiusX = int(self.scale * width / 100)
        radiusY = int(self.scale * height / 100)
        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY
        cropped = frame[minY:maxY, minX:maxX]
        resized_cropped = cv2.resize(cropped, (width, height))

        # Color detection using HUE - Source 1
        hsv = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.imshow("Frame", resized_cropped)
        cv2.imshow("Mask", mask)

        # When the ball is found - Source 1
        if len(cnts) == 1:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            self.framecounter += 1

            # Found the ball in one frame
            if self.framecounter == 1:
                self.prevX = x
                self.prevY = y

            # Calculate displacement at frame 2
            if self.framecounter == self.frame_count:
                totalX = self.prevX - x
                totalY = self.prevY - y
                totalX = float(round(totalX))
                totalY = float(round(totalY))
                Disp = (totalY ** 2 + totalX ** 2) ** 0.5
                if Disp < 0:
                    Disp *= -1
                self.times.append(self.totalsec)
                self.dispXY.append(Disp)
                datapoint = [[Disp]]
                speedC = self.linear_regressor.predict(datapoint)
                print(f"\nDisplacement: {Disp}")
                print("Linear Regression:\n", speedC)
                speed = speedC[0]
                speed = round(speed)
                self.speeds.append(speed)
                if speed > self.Maxspeed:
                    self.Maxspeed = speed
                self.framecounter = 0
                self.speedtotal += 1

                # Update GUI
                self.speed_label.config(text=f"Speed: {speed}")
                self.main_window.update()
        else:
            self.framecounter = 0

    def update_gui_final(self):
        if self.speedtotal > 0:
            self.averagespeed = sum(self.speeds) / self.speedtotal
        else:
            self.averagespeed = 0
        self.averagespeed = round(self.averagespeed)
        self.speed_label.config(text=f"Maximum Speed: {self.Maxspeed}\nAverage Speed: {self.averagespeed}")
        self.main_window.update()

    def write_results(self):
        with open('tennisdataNEW.txt', 'w') as f:
            for dp, speed, time in zip(self.dispXY, self.speeds, self.times):
                f.write(f"{dp},{speed},{time}\n")

    def compute_performance_metrics(self):
        datapoint = [[61.0]]
        predicted_speed = self.linear_regressor.predict(datapoint)
        print("\nPredicted speed for performance:\n", predicted_speed)
        print("Linear Regressor performance:")
        print("Mean absolute error =", round(sm.mean_absolute_error(self.y_test, self.y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(self.y_test, self.y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(self.y_test, self.y_test_pred), 2))
        print("Explained variance score =", round(sm.explained_variance_score(self.y_test, self.y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(self.y_test, self.y_test_pred), 2))

def main():
    tracker = BallTracker()
    tracker.process_frames()

if __name__ == "__main__":
    main()
