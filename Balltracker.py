# HEADER#########################################################################
# ANDREW PIERSON's TENNIS BALL SPEED REGRESSION MODEL
# FILENAME: Balltracker.py
# DATE: 4/23/2024
# DESCRIPTION: This python script uses OpenCV to identify a tennis ball
#               and then uses linear regression to predict the speed of
#               the ball using self collected displacement and speed data.
# INSTRUCTIONS##########################################
# 1. Run the following command LINE in terminal or run customized window.
# 2. Or simply just run the code to do live camera detection
#
# Command LINE:    python Balltracker.py --video VideoTest.mp4
# Run customized:  --video VideoTest.mp4
####THESE VALUES CAN BE ADJUSTED BASED OFF THE COLOR AND SIZE OF THE BALL###
# Scale Values:
# scale = 2
# scale = 15
scale = 25 
# scale = 33.5
# scale = 50
greenLower = (23, 80, 137) #suggestion: s-to-80 (was 60ish)
greenUpper = (83, 255, 240) #suggestion: H-to-87 V-to-240
#Sources##########################################
# Source 1- Jonathan Foster's code to using OpenCV Ball_tracking.py code to see the ball.
#           Inludes color_picker.py to test HSV values.
#   link: https://github.com/jonathanfoster/ball-tracking/blob/master/README.md
#
# Source 2- Famous Proffessor, Mark Terwiliger's linear regression code.
#   link: https://una.instructure.com/courses/86685/files/14886587?wrap=1
#
# Source 3- Stack overflow- zoom into frame code
#   link: https://stackoverflow.com/questions/50870405/how-can-i-zoom-my-webcam-in-open-cv-python
#Libraries##########################################
import numpy as np #for reading training data
from sklearn import linear_model #metrics for linear regression
import sklearn.metrics as sm # metrics for linear regression
import argparse #parsing video argument
import imutils #resizing the frame(zoom)
import cv2 #for computer vision
import matplotlib.pyplot as plt #for plotting training data
import tkinter as tk #for GUI
#CODE#########################################################################

# Gather training data - Source 2
input_file = 'tennisdata.txt'
data = np.loadtxt(input_file, delimiter=',', usecols=(0,1))
X, y = data[:, :-1], data[:, -1]
"""
# Test print - Source 2
print("X:")
print(X)
print()
print("Y:")
print(y)
print()
"""
# Training and testing for linear regression- Source 2
num_training = int(0.7 * len(X))
num_test = len(X) - num_training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
#plt.show()   #show data plot


# Command line parser for Test video - Source 1
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    print("**no video argument**")
else:
        camera = cv2.VideoCapture(args["video"])

# GUI set up - source OpenAi
main = tk.Tk()
main.geometry("750x750")
main.title("Speed Detection")
speed_label = tk.Label(main, text="Speed: 0", font=('Helvetica', 72))
speed_label.pack()

# Set fps value - no source
fpsI = camera.get(cv2.CAP_PROP_FPS)
if(25<fpsI and fpsI<35):
    fps = 30
elif (55<fpsI and fpsI<65):
    fps=60
    print(fps)

# Variables for capture and displacement
framecounter=0
max_frame_count=2
totalframes=0
times = []
totalsec=0
dispXY = []
speed=0
speeds=[]
speedC=[]
totalX=0
datapoint = [[]]
totalY=0
Disp=0
Maxspeed=0
speedtotal=0
averagespeed=0

# While taking in fames - Source 1
while True:
    # Grab video frame - Source 1
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break
    totalframes = totalframes +1

    # Calculate time in seconds - no source
    if (totalframes == fps):
        totalframes=0
        totalsec = 1+totalsec

    # Resize frame by scaling(zooming in) - source 1,3, and chatgpt
    frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]
    centerX, centerY = int(width / 2), int(height / 2)
    radiusX, radiusY = int(scale * width / 100), int(scale * height / 100)
    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY
    cropped = frame[minY:maxY, minX:maxX]
    resized_cropped = cv2.resize(cropped, (width, height))

    # Color detection using HUE - Source 1
    hsv = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.imshow("Frame", resized_cropped)
    cv2.imshow("Mask", mask)

    # CNTRs are when the ball is found - Source 1
    if (len(cnts)==1):
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        framecounter = framecounter+1

        # Found the ball in one frame  - no source
        if(framecounter == 1):
            prevX=x
            prevY=y

        # Calculate displacement at frame 2  - no source
        if (framecounter == max_frame_count):
            totalX = prevX -x
            totalY = prevY -y
            totalX=float(round(totalX))
            totalY=float(round(totalY))
            Disp = (totalY**2 + totalX**2) ** 0.5
            if(Disp<0):
                Disp=Disp*-1
            times.append(totalsec)
            dispXY.append(Disp)
            datapoint = [[Disp]]
            speedC=linear_regressor.predict(datapoint)
            print(f"\nDisplacement: {Disp}")
            print("Linear Regression:\n",speedC)
            speed = speedC[0]
            speed = round(speed)
            speeds.append(speed)
            if(speed>Maxspeed):
                Maxspeed=speed
            totalX,totalY,Disp=0,0,0
            framecounter=0
            speedtotal = speedtotal +1
            if(speed>Maxspeed):
                Maxspeed=speed

            # Update Gui - no source
            speed_label.config(text=f"Speed: {speed}")
            main.update()        
    # reset if no ball is detected
    else:
        totalX,totalY,Disp, framecounter=0,0,0,0
        
    # Quit program if q is pressed  - Source 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# Release camera and window - Source 1
camera.release()
cv2.destroyAllWindows()

# Update gui with average speed  - no source
for speed in speeds:
    averagespeed = speed +averagespeed
averagespeed = averagespeed/speedtotal
averagespeed= round(averagespeed)
speed_label.config(text=f"Maximum Speed: {Maxspeed}\nAverage Speed: {averagespeed}")
main.update()    

# Write speed and displacement to a new file  - no source
with open('tennisdataNEW.txt', 'w') as f:
    for dp,speed,time in zip(dispXY,speeds,times):
        f.write(f"{dp},{speed},{time}\n")

# Measure performance - Source 2
datapoint = [[61.0]]
speed=linear_regressor.predict(datapoint)
print("\nPredicted speed for performance:\n",speed)
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

