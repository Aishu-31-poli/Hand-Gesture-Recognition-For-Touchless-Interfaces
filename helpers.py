

import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime

class GestureVisualizer:
    
    
    @staticmethod
    def draw_hand_landmarks(frame, hand_landmarks, connections, color=(0, 255, 0)):
        """Draw hand landmarks with custom colors"""
        # Draw landmarks
        for idx, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, color, -1)
            
        # Draw connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            
            h, w, _ = frame.shape
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    @staticmethod
    def draw_gesture_info(frame, gesture_name, confidence, position=(10, 30)):
        """Draw gesture information on frame"""
        cv2.putText(frame, f"Gesture: {gesture_name}", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (position[0], position[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    @staticmethod
    def create_gesture_overlay(frame, gestures_dict):
        """Create overlay with gesture legend"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        cv2.rectangle(overlay, (w-200, 10), (w-10, 400), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw gesture list
        y_pos = 40
        for gesture_id, gesture_name in gestures_dict.items():
            cv2.putText(frame, f"{gesture_id}: {gesture_name}", 
                       (w-190, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            y_pos += 20
        
        return frame

class DataAugmentation:
    """Data augmentation for hand landmarks"""
    
    @staticmethod
    def rotate_landmarks(landmarks, angle):
        """Rotate landmarks by given angle (in degrees)"""
        # Convert to numpy array and reshape
        points = np.array(landmarks).reshape(-1, 3)
        
        # Rotation matrix
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
        
        # Apply rotation
        rotated = np.dot(points, R.T)
        
        return rotated.flatten()
    
    @staticmethod
    def scale_landmarks(landmarks, scale_factor):
        """Scale landmarks"""
        points = np.array(landmarks).reshape(-1, 3)
        scaled = points * scale_factor
        return scaled.flatten()
    
    @staticmethod
    def add_noise(landmarks, noise_level=0.01):
        """Add random noise to landmarks"""
        noise = np.random.normal(0, noise_level, len(landmarks))
        noisy = np.array(landmarks) + noise
        return noisy

class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate accuracy, precision, recall, f1-score"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def save_evaluation_report(metrics, filename):
        """Save evaluation metrics to file"""
        with open(f'logs/{filename}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f" Evaluation report saved to logs/{filename}.json")

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'utils', 'gestures', 'saved_models', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" Created directory: {directory}")

if __name__ == "__main__":
    setup_directories()
    print(" Helpers module loaded successfully!")
