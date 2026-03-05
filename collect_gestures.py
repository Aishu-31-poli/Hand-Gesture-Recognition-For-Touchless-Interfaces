

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define gestures to collect
        self.gestures = {
            0: "index_point",      # Index finger pointing - cursor movement
            1: "thumb_up",          # Thumbs up - left click
            2: "peace_sign",        # Peace sign (V) - right click
            3: "fist",              # Closed fist - scroll up
            4: "open_palm",         # Open palm - scroll down
            5: "pinch",             # Pinch gesture - drag and drop
            6: "two_finger_swipe",  # Two finger swipe - volume control
            7: "ok_sign",           # OK sign - double click
            8: "three_fingers",     # Three fingers - minimize/maximize
            9: "rock_sign"          # Rock sign - exit/close
        }
        
        self.sequence_length = 30
        self.data = []
        self.labels = []
        
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def collect_gesture(self, gesture_id, gesture_name, num_sequences=50):
        """Collect data for a specific gesture"""
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n{'='*50}")
        print(f"Collecting: {gesture_name}")
        print(f"{'='*50}")
        print("Instructions:")
        print("1. Position your hand in frame")
        print("2. Press SPACE to start recording")
        print("3. Hold the gesture steady")
        print("4. Press Q to quit early")
        print(f"\nNeed to collect: {num_sequences} sequences")
        
        sequence_count = 0
        current_sequence = []
        recording = False
        
        while sequence_count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw info on frame
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {sequence_count}/{num_sequences}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if recording:
                cv2.putText(frame, "  RECORDING...", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                               self.mp_hands.HAND_CONNECTIONS)
                    
                    if recording:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        current_sequence.append(landmarks)
                        
                        if len(current_sequence) == self.sequence_length:
                            self.data.append(current_sequence)
                            self.labels.append(gesture_id)
                            sequence_count += 1
                            current_sequence = []
                            recording = False
                            print(f"   Sequence {sequence_count} recorded")
                            time.sleep(0.5)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space
                recording = True
                current_sequence = []
                print("    Recording started...")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n Completed {sequence_count} sequences for {gesture_name}")
    
    def save_data(self):
        """Save collected data"""
        if len(self.data) > 0:
            X = np.array(self.data)
            y = np.array(self.labels)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/gesture_data_{timestamp}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump({
                    'data': X, 
                    'labels': y, 
                    'gestures': self.gestures
                }, f)
            
            print(f"\n{'='*50}")
            print(" DATA SAVED SUCCESSFULLY")
            print(f"{'='*50}")
            print(f"File: {filename}")
            print(f"Total sequences: {len(X)}")
            print(f"Data shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Gestures: {len(np.unique(y))}")
            
            return filename
        else:
            print(" No data to save")
            return None

def main():
    print("="*60)
    print(" HAND GESTURE DATA COLLECTION")
    print("="*60)
    
    collector = GestureDataCollector()
    
    print("\nAvailable gestures:")
    for idx, name in collector.gestures.items():
        print(f"  {idx}: {name}")
    
    # Collect for each gesture
    for gesture_id, gesture_name in collector.gestures.items():
        input(f"\n Press Enter to start collecting '{gesture_name}'...")
        collector.collect_gesture(gesture_id, gesture_name, num_sequences=50)
    
    # Save all data
    collector.save_data()
    
    print("\n Data collection complete!")

if __name__ == "__main__":
    main()
