'''
HandDetector.py
'''
import mediapipe as mp
import cv2


class handsDetector():
    def __init__(self, staticImageMode=False, maxNumberHands=1, modelComplexity=1, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.staticImageMode = staticImageMode
        self.maxNumberHands = maxNumberHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.staticImageMode, self.maxNumberHands, self.modelComplexity, self.minDetectionConfidence,
                                        self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=False):
        # Convert image the grayscale
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Sometimes it gets a None error so the if statement here fixes that problem
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                # for id, lm in enumerate(handLandmarks.landmark):
                    # print(id, lm)
                # Draws the connection
                if draw:
                    self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return image  

    
    '''findSnapshotWindow
    Args:
        image: the image being passed in
        draw : whether or not to draw on screen
    
    Returns:
        landmarkList: A List of all the landmarks
        window      : A list that contains
                        [0] xmin 
                        [1] ymin 
                        [2] ssW
                        [3] ssH
    '''
    def findSnapshotWindow(self, image, draw=False):
        
        window, landmarkList, xList, yList = ([] for i in range(4))
        
        offset = 25

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, landmark in enumerate(myHand.landmark):
                h, w = image.shape[:2]
                positionX = int(landmark.x * w)
                positionY = int(landmark.y * h)

                # Stores the values
                landmarkList.append([id, positionX, positionY])
                xList.append(positionX)
                yList.append(positionY)
                window = self.calculateWindow(xList, yList)
                if draw and id == 20:
                    cv2.rectangle(image, (window[0] - offset, window[1] - offset),
                                  (window[0] + window[2] + offset, window[1] + window[3] + offset),
                                  (0, 0, 255), 3)

                 
        return  landmarkList, window

    '''CalculateWindow
    Args: 
        xL: A list containing all the x positions of the joints
        yL: A list containing all the y positions of the joints
    Returns:
        window      : A list that contains
                        [0] xmin 
                        [1] ymin 
                        [2] ssW
                        [3] ssH
    '''
    def calculateWindow(self, xL, yL):
        window = []
        xmin, xmax = min(xL), max(xL)
        ymin, ymax = min(yL), max(yL)
        # print(xmin, xmax, ymin, ymax)
        # ss is short for Snapshot
        ssW, ssH = xmax - xmin, ymax - ymin
        window = xmin, ymin, ssW, ssH
        return window