## Face detection with Adaboost (Haar cascade detector)

Haar cascade detectors, based on Adaboost algorithm and OpenCV, are implemented in this programme.  
The folder '/model/cv2' includes 17 different kinds of Haar cascade detectors trained by various dataset such as frontal face, eyes, smile etc.  
Users can process images and videos. Especially, you can **import existed video** and export result video, or **process real-time video** from camera then export.

## Script Description
* *fd\_adaboost\_import.py* Import existed video and export result.
* *fd\_adaboost\_realtime.py* Process real-time video from camera without export.
* *fd\_adaboost\_realtime\_save.py* Process real-time video from camera and export.

## Environment
* Python 3.6
* OpenCV 4.2
* Python package - numpy cv2 tqdm

## Reference 
> [1] Viola P, Jones M. Rapid object detection using a boosted cascade of simple features[C]. Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001, 2001: I-I.  
> [2] Viola P, Jones M J. Robust real-time face detection[J]. International journal of computer vision, 2004, 57(2): 137-154.

## License
This code is distributed under MIT LICENSE

## Contact
xieboxuann@gmail.com