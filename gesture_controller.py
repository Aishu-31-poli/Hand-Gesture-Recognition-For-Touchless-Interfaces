import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model 

import pyautogui
import time
import json
import os
import sys
from collections import deque
import glob

class GestureController:
    def __init__(self, model_path=None):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load gesture mapping
        try:
            with open('models/gesture_mapping.json', 'r') as f:
                self.gesture_names = json.load(f)
                self.gesture_names = {int(k): v for k, v in self.gesture_names.items()}
        except:
            print(" Gesture mapping not found, using defaults")
            self.gesture_names = {
                0: "index_point", 1: "thumb_up", 2: "peace_sign",
                3: "fist", 4: "open_palm", 5: "pinch",
                6: "two_finger_swipe", 7: "ok_sign", 
                8: "three_fingers", 9: "rock_sign"
            }
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f" Model loaded from {model_path}")
        else:
            model_files = glob.glob('saved_models/final_model_*.h5')
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                self.model = load_model(latest_model)
                print(f" Model loaded from {latest_model}")
            else:
                print(" No model found! Please train first.")
                print("   Run: python train_model.py")
                sys.exit(1)
        
        # Parameters
        self.sequence_length = 30
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Control states
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5
        self.smooth_factor = 5
        self.prev_cursor_x, self.prev_cursor_y = 0, 0
        self.pinch_active = False
        self.volume_control_active = False
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.05
        
        print(f"\n Gesture Controller Ready")
        print(f" Screen: {self.screen_width} x {self.screen_height}")
        print("\nControls:")
        print("  👆 Index Point: Move cursor")
        print("  👍 Thumb Up: Left click")
        print("  ✌️ Peace Sign: Right click")
        print("  👊 Fist: Scroll up")
        print("  ✋ Open Palm: Scroll down")
        print("  🤏 Pinch: Drag & drop")
        print("  🤌 Two Fingers: Volume control")
        print("  👌 OK Sign: Double click")
        print("  🖐️ Three Fingers: Show desktop")
        print("  🤘 Rock Sign: Exit")
        print("\nPress 'Q' to quit\n")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def map_hand_to_screen(self, hand_x, hand_y):
        """Map hand position to screen coordinates"""
        screen_x = np.interp(1 - hand_x, [0, 1], [0, self.screen_width])
        screen_y = np.interp(hand_y, [0, 1], [0, self.screen_height])
        
        # Smooth movement
        if self.prev_cursor_x == 0:
            self.prev_cursor_x, self.prev_cursor_y = screen_x, screen_y
        
        smoothed_x = self.prev_cursor_x + (screen_x - self.prev_cursor_x) / self.smooth_factor
        smoothed_y = self.prev_cursor_y + (screen_y - self.prev_cursor_y) / self.smooth_factor
        
        self.prev_cursor_x, self.prev_cursor_y = smoothed_x, smoothed_y
        
        return int(smoothed_x), int(smoothed_y)
    
    def execute_action(self, gesture_id, confidence, hand_landmarks):
        """Execute action based on detected gesture"""
        current_time = time.time()
        gesture_name = self.gesture_names.get(gesture_id, "unknown")
        
        # Get hand position
        wrist = hand_landmarks.landmark[0]
        screen_x, screen_y = self.map_hand_to_screen(wrist.x, wrist.y)
        
        # Cooldown for non-continuous gestures
        if gesture_id not in [0, 5, 6]:  # Not continuous
            if current_time - self.last_gesture_time < self.gesture_cooldown:
                return True
            self.last_gesture_time = current_time
        
        # Execute actions
        if gesture_id == 0:  # Index Point - Move cursor
            pyautogui.moveTo(screen_x, screen_y)
            
        elif gesture_id == 1:  # Thumb Up - Left click
            pyautogui.click()
            print(f"   👆 Left Click ({confidence:.2f})")
            
        elif gesture_id == 2:  # Peace Sign - Right click
            pyautogui.rightClick()
            print(f"   ✌️ Right Click ({confidence:.2f})")
            
        elif gesture_id == 3:  # Fist - Scroll up
            pyautogui.scroll(3)
            print(f"   👊 Scroll Up ({confidence:.2f})")
            
        elif gesture_id == 4:  # Open Palm - Scroll down
            pyautogui.scroll(-3)
            print(f"   ✋ Scroll Down ({confidence:.2f})")
            
        elif gesture_id == 5:  # Pinch - Drag & drop
            if not self.pinch_active:
                pyautogui.mouseDown()
                self.pinch_active = True
                print(f"   🤏 Drag Started")
            else:
                pyautogui.moveTo(screen_x, screen_y)
                
        elif gesture_id == 6:  # Two Fingers - Volume
            if not self.volume_control_active:
                self.volume_control_active = True
                self.last_hand_y = wrist.y
                print(f"   🔊 Volume Control Active")
            else:
                delta_y = self.last_hand_y - wrist.y
                if abs(delta_y) > 0.03:
                    if delta_y > 0:
                        pyautogui.press('volumeup')
                        print("   🔊 Volume Up")
                    else:
                        pyautogui.press('volumedown')
                        print("   🔉 Volume Down")
                    self.last_hand_y = wrist.y
                    
        elif gesture_id == 7:  # OK Sign - Double click
            pyautogui.doubleClick()
            print(f"   👌 Double Click ({confidence:.2f})")
            
        elif gesture_id == 8:  # Three Fingers - Show desktop
            pyautogui.hotkey('win', 'd')
            print(f"   🖥️ Show Desktop ({confidence:.2f})")
            
        elif gesture_id == 9:  # Rock Sign - Exit
            print(f"   🤘 Exit Command Received")
            return False
        
        return True
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(" Cannot open camera!")
            return
        
        print(" Camera opened successfully")
        print(" Show your hand to begin...")
        
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detected_gesture = "No Hand"
            confidence = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                               self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.landmark_buffer.append(landmarks)
                    
                    # Get cursor position
                    screen_x, screen_y = self.map_hand_to_screen(
                        hand_landmarks.landmark[0].x, 
                        hand_landmarks.landmark[0].y
                    )
                    
                    # Draw cursor position
                    cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Make prediction
                    if len(self.landmark_buffer) == self.sequence_length:
                        input_data = np.array([list(self.landmark_buffer)])
                        predictions = self.model.predict(input_data, verbose=0)
                        gesture_id = np.argmax(predictions[0])
                        confidence = predictions[0][gesture_id]
                        
                        detected_gesture = f"{self.gesture_names.get(gesture_id, 'Unknown')} ({confidence:.2f})"
                        
                        # Execute action
                        if confidence > 0.7:
                            if not self.execute_action(gesture_id, confidence, hand_landmarks):
                                cap.release()
                                cv2.destroyAllWindows()
                                return
            else:
                # Reset states when no hand detected
                self.pinch_active = False
                self.volume_control_active = False
                self.landmark_buffer.clear()
            
            # Draw detected gesture
            cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw instructions
            cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Hand Gesture Controller", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n Controller stopped")

def main():
    print("="*60)
    print(" HAND GESTURE CONTROLLER")
    print("="*60)
    
    # Check if model exists
    import glob
    model_files = glob.glob('saved_models/final_model_*.h5')
    if not model_files:
        print("\n  No trained model found!")
        print("Please train the model first:")
        print("   python train_model.py")
        return
    
    # Create and run controller
    controller = GestureController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n\n Controller stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
