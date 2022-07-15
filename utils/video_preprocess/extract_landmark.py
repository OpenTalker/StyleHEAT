import face_alignment
import numpy as np
import torch


detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

mean_landmark = np.array([[264., 460.],
       [264., 540.],
       [284., 599.],
       [294., 659.],
       [313., 729.],
       [343., 798.],
       [383., 838.],
       [423., 878.],
       [512., 917.],
       [602., 897.],
       [661., 848.],
       [711., 798.],
       [751., 739.],
       [770., 659.],
       [790., 590.],
       [800., 520.],
       [810., 440.],
       [284., 401.],
       [313., 391.],
       [353., 391.],
       [393., 411.],
       [423., 431.],
       [572., 431.],
       [611., 411.],
       [651., 401.],
       [701., 401.],
       [741., 421.],
       [502., 510.],
       [492., 570.],
       [492., 619.],
       [492., 659.],
       [462., 669.],
       [472., 679.],
       [502., 689.],
       [532., 679.],
       [552., 669.],
       [343., 480.],
       [363., 470.],
       [403., 470.],
       [433., 490.],
       [403., 500.],
       [363., 500.],
       [582., 500.],
       [611., 480.],
       [651., 480.],
       [681., 490.],
       [651., 510.],
       [611., 510.],
       [423., 748.],
       [443., 739.],
       [482., 729.],
       [502., 729.],
       [522., 729.],
       [572., 739.],
       [611., 748.],
       [572., 788.],
       [542., 808.],
       [502., 818.],
       [472., 808.],
       [443., 788.],
       [423., 748.],
       [482., 748.],
       [502., 748.],
       [532., 748.],
       [602., 748.],
       [532., 768.],
       [502., 778.],
       [472., 778.]])

mean_landmark /= 4

def get_landmark(images):
    """
    :param images: PIL list
    :return: numpy list
    """
    # if isinstance(images, list):
    #     lm = []
    #     for i in images:
    #         lm.append(detector.get_landmarks_from_image(np.array(images))[0])
    # else:
    lms_np = []
    for image in images:
        try:
            lm = detector.get_landmarks_from_image(np.array(image))[0]
            lms_np.append(lm)
        except:
            lm = mean_landmark
            lms_np.append(lm)
    lms_np = np.stack(lms_np)  # B, 68, 2
    return lms_np
