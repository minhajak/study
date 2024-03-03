import cv2 as cv
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as drawing


#get hand landmark
def getHandLandmarks(img, draw):
    lmlist = []

    hands = mpHands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)#BGR to RGB
    
    handsDetected = hands.process(frameRGB)
    if handsDetected.multi_hand_landmarks:
        for landmarks in handsDetected.multi_hand_landmarks:
            # print(landmarks)
            for id, lm in enumerate(landmarks.landmark):
                # print(id, lm)
                h,w,c  =img.shape#height, width, channels
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append((id, cx, cy))
                # print(lmlist)

        if draw:
            drawing.draw_landmarks(
                img, 
                landmarks,
                mpHands.HAND_CONNECTIONS
            )
    return lmlist


#finger counting
def fingerCount(lmlist):
    count = 0
    if lmlist[8][2] < lmlist[6][2]:
        count += 1

    if lmlist[12][2] < lmlist[10][2]:
        count += 1

    if lmlist[16][2] < lmlist[14][2]:
        count += 1

    if lmlist[20][2] < lmlist[18][2]:
        count += 1

    if lmlist[4][1] < lmlist[2][1]:
        count += 1

    return count



    
#setup the camera
cam = cv.VideoCapture(0)# 0 - build in camera 

while True:
    success, frame = cam.read()
    if not success:
        print('camera not detected')
        

    # frame = cv.flip(frame,1)# flip to adjust the camera to right side
    lmlist = getHandLandmarks(img=frame, draw=False )

    if lmlist:
        # print(lmlist)
        fc = fingerCount(lmlist=lmlist)
        # print(fc)
        cv.rectangle(frame, (400,10), (600,250), (0,0,0), -1)
        cv.putText(frame, str(fc), (400,255),cv.FONT_HERSHEY_PLAIN, 20, (0,255,255),30)
           
    cv.imshow('Ai Frame Counting', frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()