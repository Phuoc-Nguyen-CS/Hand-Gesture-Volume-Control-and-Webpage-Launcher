
# Hand Gesture Volume Control and Webpage Launcher

This program uses OpenCV, TensorFlow, and Google's MediaPipe to detect hand gestures and perform certain actions based on the detected gesture. The actions include muting/unmuting the volume, raising/lowering the volume, and opening certain common web pages on Google for the user.

## Installation

### Github
1.  Clone the repository using the command below:

    `git clone https://github.com/Phuoc-Nguyen-CS/Hand-Gesture-Volume-Control-and-Webpage-Launcher.git` 

2.  Install the required libraries using the command below:

    `pip install -r requirements.txt` 

### Exe File
1. Go to: "https://drive.google.com/drive/folders/15Ymnuz33PyBSVSkDqS5YzpqU2l09IxnR?usp=sharing"
2. Download the file!
3. Run main.exe

## Note

If you want to train your own tensorflow model, follow these steps.

Setup:
* Create a "DataCollection" folder within the root folder
* Create subfolders from 0 to 9 within that folder.

Data Collection:
* Launch the program
* Press "s" on your keyboard to enter "Snapshot Mode".
* Use "0" through "9" to save the images within the correct folder.

Data Training:
> **_NOTE:_** There are many ways to train, but I found this to be the easiest.
* Go to: ```https://teachablemachine.withgoogle.com/train/image```
* Follow the instructions
* Export the data in Tensorflow Keras Model.
* Replace the model within TrainedModel folder

## Usage

1.  Run the program using the command below:

    `python main.py` 

2.  Place your hand in front of the camera and perform one of the following gestures:
    > **_NOTE:_** Some of these commands at the moment are configured to different hand gestures. I still need to retrain my data a bit
    -   "Mute": Touch the tip of your thumb and pointer finger together and bring all your other fingers to match your pointer finger's level.
    -   "Unmute": Hold your hand open with your palm facing forward.
    -   "Raise volume": Point up.
    -   "Lower volume": Point down.
    -   "Launch Google": Make a Peace Sign (tentative to change)
    -   "Launch Youtube:" Point at the Screen (tentative to change)

3.  The program will perform the action based on the detected gesture.

## Credits

-   [OpenCV](https://opencv.org/)
-   [TensorFlow](https://www.tensorflow.org/)
-   [MediaPipe](https://google.github.io/mediapipe/)
-   [Comtypes](https://pythonhosted.org/comtypes/)
-   [Pycaw](https://pypi.org/project/pycaw/)
-   [Webbrowser](https://docs.python.org/3/library/webbrowser.html)
