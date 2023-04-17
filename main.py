import cv2
import mediapipe as mp
import HandsDetector
import numpy as np


''' selectMode
    Args: 
        key: the key being pressed
        screenshot: a boolean that dictates if in screenshotting mode
    Returns: 
        number: O-9 on the keyboard
                -1 if no number               
        screenshot: a boolean that would be reversed to turn on and off screenshotting
'''
def selectMode(key, screenshot):
    # Give an out of bound number indicating no number in bound was pressed
    number = -1

    if key == ord('s'):
        # print(screenshot)
        return number, not screenshot
    
    if (screenshot == True) and (ord('0') <= key <= ord('9')):
        number = key - ord('0')
        # print(number)    
        return number, screenshot

    return number, screenshot    

if __name__ == "__main__":
    # Set the default
    cameraWidth = 640  
    cameraHeight = 480
    
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(3, cameraWidth)
    capture.set(4, cameraHeight)


    detector = HandsDetector.handsDetector()

    screenshot = False
    offset = 25
    imageSize = 400
    # Keeps the camera open and runs all the necessary calculations
    while True:
        
        key = cv2.waitKey(1)
        number, screenshot = selectMode(key, screenshot)
        # print(number, screenshot)
        # if 0 <= number <= 9:
        #     print("Thank you") 

        # Break out of the loop using "ESC"
        if key == 27:
            break
        # Closes out the window once done with snapshot mode
        elif key == ord('s') and screenshot == False:
            cv2.destroyWindow("HandCam")
            cv2.destroyWindow("SnapshotView")

        success, image = capture.read()
        # Mirrors the image so I don't lose my mind
        # image = cv2.flip(image, 1)
        image = detector.findHands(image, draw=False)

        landmarkList, window = detector.findSnapshotWindow(image, draw=True)

        if screenshot and len(landmarkList) != 0:
            try:
                whiteImage = np.ones((imageSize, imageSize, 3), np.uint8) * 255
                croppedImage =  image[window[1]-offset:window[1]+window[3]+offset,
                                    window[0]-offset:window[0]+window[2]+offset]
                
                croppedShape = croppedImage.shape
                if croppedShape[0] <= imageSize and croppedShape[1] <= imageSize:
                    whiteImage[0:croppedShape[0], 0:croppedShape[1]] = croppedImage
                    cv2.imshow("HandCam", croppedImage)
                    cv2.imshow("SnapshotView", whiteImage)
                else:
                    print("[Error]: croppedImage > whiteImage")
            except cv2.error as error:
                # x/y is at 0. out of bounds
                print("[Error]: {}".format(error))

        cv2.imshow("Gesture Control", image)

    # Cleanly closes
    capture.release()
    cv2.destroyAllWindows()


