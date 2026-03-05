

import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import os
import glob
import datetime
import matplotlib.pyplot as plt

class GestureTrainer:
    def __init__(self):
        self.model = None
        self.sequence_length = 30
        self.feature_count = 63  # 21 landmarks * 3 (x,y,z)
        self.num_classes = 10
        self.gesture_names = {
            0: "index_point",
            1: "thumb_up",
            2: "peace_sign",
            3: "fist",
            4: "open_palm",
            5: "pinch",
            6: "two_finger_swipe",
            7: "ok_sign",
            8: "three_fingers",
            9: "rock_sign"
        }
        
    def load_data(self):
        """Load all collected data"""
        files = glob.glob("data/gesture_data_*.pkl")
        
        if not files:
            print(" No data files found in data/ folder!")
            return None, None
        
        all_data = []
        all_labels = []
        
        print(f"\n Found {len(files)} data files")
        for file in files:
            print(f"   Loading: {file}")
            with open(file, 'rb') as f:
                data_dict = pickle.load(f)
                all_data.append(data_dict['data'])
                all_labels.append(data_dict['labels'])
        
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        print(f"\n Loaded {len(X)} sequences")
        print(f"   Data shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        return X, y
    
    def create_model(self):
        """Create CNN-LSTM model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=(self.sequence_length, self.feature_count)),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            LSTM(units=128, return_sequences=True),
            Dropout(0.3),
            
            LSTM(units=64, return_sequences=False),
            Dropout(0.3),
            
            Dense(units=64, activation='relu'),
            Dropout(0.3),
            
            Dense(units=32, activation='relu'),
            
            Dense(units=self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\n Model created")
        print(model.summary())
        
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        
        # Load data
        X, y = self.load_data()
        if X is None:
            return
        
        # Preprocess
        X = X.astype(np.float32)
        y_encoded = to_categorical(y, num_classes=self.num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, 
            stratify=np.argmax(y_encoded, axis=1)
        )
        
        print(f"\n Training samples: {len(X_train)}")
        print(f" Testing samples: {len(X_test)}")
        
        # Create model
        self.create_model()
        
        # Callbacks
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = ModelCheckpoint(
            f'saved_models/best_model_{timestamp}.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        print("\n Starting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"\n{'='*50}")
        print(f" Test Accuracy: {test_accuracy:.4f}")
        print(f" Test Loss: {test_loss:.4f}")
        print(f"{'='*50}")
        
        # Save final model
        final_model_path = f'saved_models/final_model_{timestamp}.h5'
        self.model.save(final_model_path)
        print(f"\n Model saved to: {final_model_path}")
        
        # Save gesture mapping
        import json
        with open('models/gesture_mapping.json', 'w') as f:
            json.dump(self.gesture_names, f)
        print(" Gesture mapping saved")
        
        # Plot results
        self.plot_training_history(history, timestamp)
        
        return history, test_accuracy
    
    def plot_training_history(self, history, timestamp):
        """Plot training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'logs/training_history_{timestamp}.png')
        plt.show()

def main():
    print("="*60)
    print(" HAND GESTURE MODEL TRAINING")
    print("="*60)
    
    trainer = GestureTrainer()
    trainer.train(epochs=50, batch_size=32)
    
    print("\n Training complete!")

if __name__ == "__main__":
    main()
