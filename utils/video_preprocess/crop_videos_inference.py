import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import numpy as np
from PIL import Image
import dlib

from torch.multiprocessing import Pool, Process, set_start_method


class Croper:
    # Used for video, output size should be
    def __init__(self):
        pass

    def align_face(self, img, lm, output_size=256):
        """
        :param filepath: str
        :return: PIL Image
        """
        # lm_chin = lm[0: 17]  # left-right
        # lm_eyebrow_left = lm[17: 22]  # left-right
        # lm_eyebrow_right = lm[22: 27]  # left-right
        # lm_nose = lm[27: 31]  # top-down
        # lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        # lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        transform_size = 1024
        enable_padding = False

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        # if output_size < transform_size:
        #     img = img.resize((output_size, output_size), Image.ANTIALIAS)

        # Transform.
        def align_func(img):
            img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
            if output_size < transform_size:
                img = img.resize((output_size, output_size), Image.ANTIALIAS)
            return img

        # # Transform.
        # quad = (quad + 0.5).flatten()
        # lx = max(min(quad[0], quad[2]), 0)
        # ly = max(min(quad[1], quad[7]), 0)
        # rx = min(max(quad[4], quad[6]), img.size[0])
        # ry = min(max(quad[3], quad[5]), img.size[0])
        # img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(),
        #                     Image.BILINEAR)
        # if output_size < transform_size:
        #     img = img.resize((output_size, output_size), Image.ANTIALIAS)

        # Save aligned image.
        return align_func

    def crop(self, frame_pils, lm_np):
        func = self.align_face(img=frame_pils[0], lm=lm_np[0], output_size=256)
        os.makedirs('temp', exist_ok=True)
        for _i in range(len(frame_pils)):
            frame_pils[_i] = func(frame_pils[_i])
            # func(frame_pils[_i]).save(f'./temp/{_i}.jpg')

        # lm = self.get_landmark(img_np)
        # if lm is None:
        #     return None
        # crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=512)
        # clx, cly, crx, cry = crop
        # lx, ly, rx, ry = quad
        # lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # for _i in range(len(img_np_list)):
        #     _inp = img_np_list[_i]
        #     _inp = _inp[cly:cry, clx:crx]
        #     _inp = _inp[ly:ry, lx:rx]
        #     img_np_list[_i] = _inp
        return frame_pils
