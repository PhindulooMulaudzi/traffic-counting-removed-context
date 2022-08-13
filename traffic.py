import os

import random
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import utils as myutils #this is a python file containing support functions
import skvideo
skvideo.setFFmpegPath('C:/ffmpeg')
import skvideo.io

# a few problematic interactions between the python bindings and OpenCL
# disabling OpenCL support in run-time to avoid problems
cv2.ocl.setUseOpenCL(False)
random.seed(123)

from pipeline import (
PipelineRunner,
ContourDetection,
Visualizer,
CsvWriter,
VehicleCounter
)

Image_dir = "./out"
Video_source = "road.mp4"
Shape = (720,1280)  #height*width
Exit_pts = np.array([
                    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
                    [[0, 400], [645, 400], [645, 0], [0, 0]]
                    ])

def train_bg_subtractor(bg_object, cap, num=500):
    '''
    BG subtractor needs to process some number of frames to start giving results
    '''
    print("Training BG subtractor! ")
    i = 0
    for frame in cap:
        bg_object.apply(frame, None, 0.001)
        i += 1
        if i>= num:
            return cap


def main():
    # creating an exit mask from points where we will be counting out vehicles
    base = np.zeros(Shape + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, Exit_pts, (255,255,255))[:,:,0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True)

    # processing pipeline for programming convenience
    pipeline = PipelineRunner(pipeline=[
    ContourDetection(bg_subtractor=bg_subtractor, save_image=False, image_dir=Image_dir),
    # we use y_weight == 2.0 because traffic are moving vertically on video
    # use x_weight == 2.0 for horizontal
    VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
    Visualizer(image_dir=Image_dir),
    CsvWriter(path='./', name='report.csv')
    ])

    #setting up image source
    cap = skvideo.io.vreader(Video_source)

    # skip 500 frames to train bg subtractor
    # instead of skipping frames we will have a separate video for training
    # this is per my inital thoughts on the proposal
    train_bg_subtractor(bg_subtractor, cap, num=50)
    _frame_number = -1
    frame_number = -1

    # lets start open cv capture and then can open the window
    capVideo = cv2.VideoCapture()
    capVideo.open(Video_source)

    if not capVideo.isOpened():
        print("error reading video file")

    while capVideo.isOpened():
        # lets get the frame, this is the frame that will be sent to the pipeline
        _, frame = capVideo.read()

        # real frame number
        _frame_number += 1
        #skipping every 2nd frame to speed up processing
        if _frame_number % 2!=0:
            continue
        frame_number += 1

        pipeline.run(frame, frame_number)

        if cv2.waitKey(25) & 0xff == 27:
            cv2.destroyAllWindows()
            break

        if ((capVideo.get(cv2.CAP_PROP_POS_FRAMES) + 1) > capVideo.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

    capVideo.release()
    cv2.destroyAllWindows()

# we check if the out image directory exists or not and call main function
if __name__ == "__main__":
    if not os.path.exists(Image_dir):
        os.makedirs(Image_dir)
    main()