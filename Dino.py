import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import math
keyboard = Controller()
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def findDistance(p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        #if img is not None:
            #cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            #cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            #cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            #cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img
while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(imgRGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id,lm in enumerate(handLms.landmark):
                h,w,c = imgRGB.shape
                cx,cy,cz = int(lm.x*w),int(lm.y*h),int(lm.z*w)
                lmList.append([cx,cy,cz])
            

            #mp_drawing.draw_landmarks(img,handLms,mp_hands.HAND_CONNECTIONS)
        
        length,_,img = findDistance((lmList[8][0],lmList[8][1]),(lmList[4][0],lmList[4][1]),img,(255,0,0),2)
        if length<25:
                                              
             keyboard.press(Key.space)
             keyboard.release(Key.space)
            
        
    cv2.imshow('image',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
