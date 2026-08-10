import mediapipe as mp
import cv2
import time
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

#Webcam setup
cap = cv2.VideoCapture(0)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#Initialize hands object from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)


while (True):
        ret, frame = cap.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                for point in handsModule.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]
                    pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
                    cv2.circle(frame, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
        cv2.imshow('Test hand', frame)
        if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
cap.release()