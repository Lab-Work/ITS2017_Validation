# Traffic Estimation through a Side-View Camera

This is the official repository containing the script and a sample video data for vehicle detection and speed estimation in a side-view traffic camera. The script is written in Python and has dependency on Scipy, OpenCV, and scikit-learn. The data was collected on Neil Street on May 30th, 2017 in Urbana, IL, where a total of 55 vehicles were recorded in the video. This is part of the research conducted with Yanning Li and Prof. Daniel B. Work at the Coordinated Science Lab at UIUC.

## Installation 
First install the following python depencies:
```
pip install numpy, matplotlib, scipy, scikit-learn
```
Next, install or compile OpenCV Python binding on your system. For Ubuntu 16.04 users, you may follow the instuction [here](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/).

## Usage
To process the sample video data, 
```
python run.py
```

To customize the script to a different side-view traffic footage, one needs to measure the following items:
+ Frame per second
+ Horizontal field of view (rad)
+ Camera rotation in the plane perpendicular to the lens center axis (rad)
+ Pixel distances of three horizontal lines in the frame with respect to the image origin (pixel)
+ Physical distances of these three horizontal line with respect to the camera position (m)

## Extended Dataset
More video data can be downloaded [here](https://uofi.box.com/s/i3ac71aejupj1khbo7w61az9qe6u5tz2). The linked Box folder contains scripts to process each individual video footage.

## Contact
Author: Fangyu Wu, Coordinated Science Laboratory, UIUC

Email: fwu10(at)illinois.edu

Web: fangyuwu.com
