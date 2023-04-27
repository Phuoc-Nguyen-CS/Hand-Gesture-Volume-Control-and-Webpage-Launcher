# Imports to deal with all the detection and datacollection
import os
import cv2
import mediapipe as mp
import HandsDetector
import numpy as np
import math
import uuid
from ClassificationModule import Classifier

# Imports for volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Imports for web control
import webbrowser

# Global Values
offset = 25
imageSize = 400
waitPredict = 0

gestureName = {
    0 : 'Mute',
    1 : 'Palms Open',
    2 : 'Heart',
    3 : 'Palms Closed',
    4 : 'Peace',
    5 : 'Point'
}

dataCollectionList = [
    'DataCollection/0',
    'DataCollection/1',
    'DataCollection/2',
    'DataCollection/3',
    'DataCollection/4',
    'DataCollection/5',
    'DataCollection/6',
    'DataCollection/7',
    'DataCollection/8', 
    'DataCollection/9'
]

# Audio Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Web Setup
url = 'https://www.google.com'
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))

''' selectMode
    Args: 
        key: the key being pressed.
        snapshot: a boolean that dictates if in screenshotting mode.
        ls: a boolean that dictates if the program is in listing mode.
        watch: a boolean that dictates if the program is watching for inputs.
    Returns: 
        number: O-9 on the keyboard
                -1 if no number               
        snapshot: a boolean that would be reversed to turn on and off screenshotting.
        ls : a boolean that would be reversed to turn on and off listing mode.
'''
def selectMode(key, snapshot, ls, watch):
    # Give an out of bound number indicating no number in bound was pressed
    number = -1
    folder = ''
    # ==========================================================

    # Go into/out snapshot mode
    if ((key == ord('s')) or (key == ord('S'))):
        if snapshot == False:
            print('Snapshot mode: Activated')
        else:
            print('Snapshot mode: Deactivated')
        return folder, not snapshot, ls, watch

    # Go into/out ls mode
    elif ((key == ord('l')) or (key == ord('L'))):
        if ls == False:
            print('Listing Mode: Activated')
        else:
            print('Listing Mode: Deactivated')
        return folder, snapshot, not ls, watch

    # Go into watch mode
    elif ((key == ord('w')) or (key == ord('W'))):
        if watch == False:
            print('Watch Mode: Activated')
        else:
            print('Watch Mode: Deactivated')
        return folder, snapshot, ls, not watch
    
    # ==========================================================
    
    # Prints the number of snapshots within the given folder
    if (ls == True) and (ord('0') <= key <= ord('9')):
        number = key - ord('0')
        dirPath = folder + str(number)
        numFiles = 0

        # Go through the  directory
        for images in os.listdir(dirPath):
            # Checking if file
            if os.path.isfile(os.path.join(dirPath, images)):
                numFiles += 1
        
        print(numFiles)
        return folder, snapshot, ls, watch

    # Returns the folder where to store the snapshot
    if (snapshot == True) and (ord('0') <= key <= ord('9')):
        number = key - ord('0')
        folder = 'DataCollection/'
        folder += str(number)  
        return folder, snapshot, ls, watch

    return folder, snapshot, ls, watch

''' audioControl
    Args:
        name: The gestureName[index] being passed in to compare to the gestureName dictionary
    Returns:
        Nothing
'''
def audioControl(name):
    # Mute: 
    if name == gestureName[0]:
        volume.SetMute(1, None)
    # Unmute:
    elif name == gestureName[1]:
        volume.SetMute(0, None)
    # Volume Up
    elif name == gestureName[2]:
        currentVolume = volume.GetMasterVolumeLevelScalar()
        # Rounding to nearest 10
        roundedVolume = round(currentVolume, 1)
        print('RoundedVolume in raise volume = ' + str(roundedVolume))
        try:
            volume.SetMasterVolumeLevelScalar(roundedVolume + .10, None)
        except:
            print('Volume is already at max')
    # Volume Down
    elif name == gestureName[3]:
        currentVolume = volume.GetMasterVolumeLevelScalar()
        roundedVolume = round(currentVolume, 1)
        print('RoundedVolume in decrease volume = ' + str(roundedVolume))
        # Rounding to nearest 10
        try:
            volume.SetMasterVolumeLevelScalar(roundedVolume + -.10, None)
        except:
            print('Volume is already at min')

'''webControl
    Args:
        name: The gestureName[index] being passed in to compare to the gestureName dictionary
    Returns:
        Nothing
'''
def webControl(name):
    if name == gestureName[4]:
        webbrowser.get('chrome').open(url)
    elif name == gestureName[5]:
        webbrowser.get('chrome').open('https://www.youtube.com')

''' watchMode
    Args:
        image: The current frame being passed in
        detector: Google's Mediapipe hand detector
        classifier: CVZone's Class to help with tensorflow prediction with the given hand gesture
    Returns:
        Nothing
'''
def watchMode (image, detector, classifier):
    
    global waitPredict 
    landmarkList, window = detector.findSnapshotWindow(image, draw=True)

    # If a hand is detected within the frame
    if len(landmarkList) != 0:
        try:
            croppedImage =  image[window[1]-offset:window[1]+window[3]+offset,
                                    window[0]-offset:window[0]+window[2]+offset]
            croppedShape = croppedImage.shape
            cv2.imshow("HandCam", croppedImage)

            waitPredict += 1

            if waitPredict == 25:
                prediction, index = classifier.getPrediction(croppedImage, draw=False)
                print(gestureName[index])
                # First 4 control the audio
                if index >= 0 and index <= 3:
                   audioControl(gestureName[index])
                # Remaining controls the web
                elif index >= 4 and index <= 5:
                   webControl(gestureName[index]) 
                waitPredict = 0      
        except cv2.error as error:
            # x/y is at below 0. out of bounds
            print("[Error]: {}".format(error)) 

''' snapshotMode
    Args:
        image: The current frame being passed in
        detector: Google's Mediapipe hand detector
        folder: The folder name being passed in to save the data to
    Returns:
        Nothing
'''
def snapshotMode (image, detector, folder):
    landmarkList, window = detector.findSnapshotWindow(image, draw=True)

    if len(landmarkList) != 0:
        try:
            # Convert to 8 bit image using np.ones
            # An image of imageSize with RGB = 8 bit image 
            # By multipling 255 we are able to get the color
            whiteImage = np.ones((imageSize, imageSize, 3), np.uint8) * 255
            croppedImage =  image[window[1]-offset:window[1]+window[3]+offset,
                                window[0]-offset:window[0]+window[2]+offset]
            croppedShape = croppedImage.shape

            imageRatio = window[3] / window [2]
            
            # Height > Width
            # Condense the Width to accomodate for the height to fit the image
            if imageRatio > 1:
                
                # Get the total image size and divide it by the ssH
                value = imageSize / window[3]
                newWidth = math.ceil(value * window[2])
                resizedImage = cv2.resize(croppedImage, (newWidth, imageSize))
                resizedShape = resizedImage.shape
                leftGap = math.ceil((imageSize - newWidth) / 2)

                if resizedShape[0] <= imageSize and resizedShape[1] <= imageSize:
                # We want to select all rows ':'
                # We then want to use the slice we just created 
                # leftGap -> slicedImageTotal
                # Place the image within that slice
                    whiteImage[:, leftGap: newWidth + leftGap] = resizedImage

            # Width > Height
            # Condense the Height to accomodate for the width to fit the image
            else:        
                # Get the total image size and divide it by the ssW
                value = imageSize / window[2]
                newHeight = math.ceil(value * window[3])
                resizedImage = cv2.resize(croppedImage, (newHeight, imageSize))
                resizedShape = resizedImage.shape
                heightGap = math.ceil((imageSize - newHeight) / 2)

                if resizedShape[0] <= imageSize and resizedShape[1] <= imageSize:
                # We want to select all rows ':'
                # We then want to use the slice we just created 
                # heightGap -> slicedImageTotal
                # Place the image within that slice
                    whiteImage[:, heightGap: newHeight + heightGap] = resizedImage
            cv2.imshow('SnapshotView', whiteImage)
        except cv2.error as error:
            # x/y is at below 0. out of bounds
            print('[Error]: {}'.format(error))

    if any(folder == dataCollection for dataCollection in dataCollectionList):
        picture = cv2.imwrite(f'{folder}/{str(uuid.uuid4())}.jpg', whiteImage)

''' main
    The main program where most of the OpenCV is done.
'''
def main():
        
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Set the default
    cameraWidth = 640  
    cameraHeight = 480

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(3, cameraWidth)
    capture.set(4, cameraHeight)

    detector = HandsDetector.handsDetector()
    classifier = Classifier('TrainedModel/keras_model.h5', 'TrainedModel/labels.txt')
    snapshot = False
    ls = False
    watch = False

    # === Program Starts Here =====================================

    while True:
        
        # Receives key value from keyboard
        key = cv2.waitKey(1)
        folder, snapshot, ls, watch = selectMode(key, snapshot, ls, watch)
 
        # Break out of the loop using 'ESC'
        if key == 27:
            break
        # Closes out the window once done with snapshot mode
        elif ((key == ord('s')) or (key == ord('S')))  and snapshot == False:
            if cv2.getWindowProperty('SnapshotView', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('SnapshotView')
        elif ((key == ord('w')) or (key == ord('W'))) and watch == False:
            cv2.destroyWindow('HandCam')

        # Reads the camera
        success, image = capture.read()
        # Mirrors the image so I don't lose my mind
        image = cv2.flip(image, 1)
        image = detector.findHands(image, draw=False)

        # Find the snapshot values and the list of landmarks for the hands
        landmarkList, window = detector.findSnapshotWindow(image, draw=True)
        
        # === Watch Mode ==========================================================
        if watch:
            watchMode(image, detector, classifier)
        
        # === Snapshot Mode =======================================================
        if snapshot:
            snapshotMode(image, detector, folder)

        cv2.imshow('Gesture Control', image)

    # Cleanly closes
    capture.release()
    cv2.destroyAllWindows()
    print('Cleanly closed the program')

if __name__ == "__main__":
    main()
    