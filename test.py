import os 
import cv2 
import numpy as np
import mediapipe as mp
import joblib 

#IMPORTANT IF UR MODEL IS DIFFERENT/IS IN A DIFFERENT FOLDER MAKE SURE TO CHANGE THIS PATH TO WHEREVER YOUR MODEL IS LOCATED 
#EX. Model_path = os.path.join("models", "your_model_name.pkl")
Model_path = os.path.join("models", "randomForest_model.pkl")

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    wrists = coords[0].copy()
    coords -= wrists
    max_dist= np.linalg.norm(coords, axis=1).max()
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()

def main(): #setsup openCV and mediapipe USE CNTRL C to close the window also might take a bit to load
    if not os.path.exists(Model_path):
        print(f"Model file not found at {Model_path}")
        return
    model = joblib.load(Model_path)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_landmarks = normalize_landmarks(hand_landmarks)
                prediction = model.predict([normalized_landmarks])
                label = prediction[0]
                cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()