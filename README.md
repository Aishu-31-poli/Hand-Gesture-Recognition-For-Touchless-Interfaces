

HandGestureProject
│

STEP 1: CREATE AND ACTIVATE VIRTUAL ENVIRONMENT
# Create virtual environment with Python 3.10

-->  py -3.10 -m venv venv

# Activate virtual environment (Windows)
-->  venv\Scripts\activate


STEP 2: INSTALL REQUIRED PACKAGES
# Install all dependencies from requirements.txt
-->  pip install -r requirements.txt

# If you don't have requirements.txt, install manually:

pip install tensorflow==2.13.0
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.7
pip install numpy==1.24.3
pip install pyautogui==0.9.54
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install pandas==2.0.3
pip install keyboard==0.13.5
pip install pillow==10.0.0
pip install protobuf==3.20.3


STEP 3: COLLECT DATASET
# Make sure (venv) is activated
-->  python collect_dataset.py



Gestures to Collect:

No.	Gesture Name	        Action             

1	index_point 	        Move cursor               
2	thumb_up	        Left click
3	peace_sign	        Right click
4	fist	                Scroll up
5	open_palm	        Scroll down
6	pinch	                Drag & drop
7	two_finger_swipe	Volume control
8	ok_sign	                Double click
9	three_fingers	        Show desktop
10	rock_sign	        Exit program

      
STEP 4: TRAIN MODEL
# Make sure (venv) is activated
-->  python train.py



STEP 5: REAL-TIME RECOGNITION
# Make sure (venv) is activated
-->  python realtime.py

final output


        👆 Index Point: Move cursor
        👍 Thumb Up: Left click
        ✌️ Peace Sign: Right click
        👊 Fist: Scroll up
        ✋ Open Palm: Scroll down
        
        🤏 Pinch: Drag & drop
        🤌 Two Fingers: Volume control
        👌 OK Sign: Double click
        🖐️ Three Fingers: Show desktop
        🤘 Rock Sign: Exit