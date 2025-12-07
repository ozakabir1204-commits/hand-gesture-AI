import cv2
import mediapipe as mp
import numpy as np
import random

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720
COLOR_DEFAULT = (255, 0, 255)  # Purple
COLOR_GRABBED = (0, 255, 0)    # Green

# --- CLASS FOR VIRTUAL OBJECTS ---
class DragObject:
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size
        self.color = COLOR_DEFAULT

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if finger is inside the box
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.color = COLOR_GRABBED
            self.posCenter = cursor  # Move object to finger position
        else:
            self.color = COLOR_DEFAULT

# --- INITIAL SETUP ---
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Create a list to hold our virtual objects
rectList = []
# Create 3 initial objects
for x in range(3):
    rectList.append(DragObject([x * 250 + 150, 150]))

print("--- CONTROLS ---")
print("1. Pinch (Index + Thumb) to grab an object.")
print("2. Press 'a' on your keyboard to ADD a new object.")
print("3. Press 'q' to QUIT.")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip image so it acts like a mirror
    img = cv2.flip(img, 1)
    
    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    cursor = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw skeleton on hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Get coordinates of Index Finger (8) and Thumb (4)
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if lmList:
                # Tip of Index (8) and Thumb (4)
                x1, y1 = lmList[8][1], lmList[8][2]
                x2, y2 = lmList[4][1], lmList[4][2]
                
                # Calculate distance between fingers
                length = np.hypot(x2 - x1, y2 - y1)
                
                # If distance is short, we are "pinching" / "clicking"
                if length < 40:
                    # Draw a circle to show we are clicking
                    cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 15, (0, 255, 0), cv2.FILLED)
                    cursor = [x1, y1] # The "mouse" position is the index finger

    # --- UPDATE OBJECTS ---
    # Create a semi-transparent layer for cool visuals
    imgNew = np.zeros_like(img, np.uint8)
    
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        
        # If we are pinching, try to move the object
        if cursor:
            rect.update(cursor)
            
        # Draw the rectangle
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), rect.color, cv2.FILLED)

    # Combine the transparent layer with the real webcam image
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Show instructions on screen
    cv2.putText(out, "Pinch to Drag | Press 'a' to Add | 'q' to Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Mr. Oza's AI Hand Controller", out)
    
    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1)
    
    # Press 'a' to ADD a new random object
    if key == ord('a'):
        rand_x = random.randint(100, WIDTH - 100)
        rand_y = random.randint(100, HEIGHT - 100)
        rectList.append(DragObject([rand_x, rand_y]))
        print(f"Added new object at {rand_x}, {rand_y}")
        
    # Press 'q' to QUIT
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()