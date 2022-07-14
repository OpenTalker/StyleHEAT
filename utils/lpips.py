import os
import glob
import numpy as np
from imageio import imread

import torch
from third_part.PerceptualSimilarity.models import dist_model as dm
from utils.distributed import master_only_print as print


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list' % flist)
    return []


def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list = []
    distorated_list = []

    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)
        image = image.split('_2_')[-1]
        image = image.split('_vis')[0] + '.jpg'
        gt_image = os.path.join(gt_path, image)
        if not os.path.isfile(gt_image):
            gt_image = gt_image.replace('.jpg', '.png')
        gt_list.append(gt_image)
        distorated_list.append(distorted_image)
    return gt_list, distorated_list


class LPIPS():
    def __init__(self, use_gpu=True):
        self.model = dm.DistModel()
        self.model.initialize(model='net-lin', net='alex', use_gpu=use_gpu)
        self.use_gpu = use_gpu

    def __call__(self, image_1, image_2):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        result = self.model.forward(image_1, image_2)
        return result

    def calculate_from_disk(self, gt_path, distorted_path, batch_size=64, verbose=False, for_deformation=True):
        # if sort:
        if for_deformation:
            files_1, files_2 = preprocess_path_for_deform_task(gt_path, distorted_path)
        else:
            files_1 = sorted(get_image_list(gt_path))
            files_2 = sorted(get_image_list(distorted_path))

        new_files_1, new_files_2 = [], []
        for item1, item2 in zip(files_1, files_2):
            if os.path.isfile(item1) and os.path.isfile(item2):
                new_files_1.append(item1)
                new_files_2.append(item2)
            else:
                print(item2)
        imgs_1 = np.array([imread(str(fn)).astype(np.float32) / 127.5 - 1 for fn in new_files_1])
        imgs_2 = np.array([imread(str(fn)).astype(np.float32) / 127.5 - 1 for fn in new_files_2])

        # Bring images to shape (B, 3, H, W)
        imgs_1 = imgs_1.transpose((0, 3, 1, 2))
        imgs_2 = imgs_2.transpose((0, 3, 1, 2))

        result = []

        d0 = imgs_1.shape[0]
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size
        n_used_imgs = n_batches * batch_size

        # imgs_1_arr = np.empty((n_used_imgs, self.dims))
        # imgs_2_arr = np.empty((n_used_imgs, self.dims))
        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                # end='', flush=True)
            start = i * batch_size
            end = start + batch_size

            img_1_batch = torch.from_numpy(imgs_1[start:end]).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2[start:end]).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()

            result.append(self.model.forward(img_1_batch, img_2_batch))

        distance = np.average(result)
        print('lpips: %.3f' % distance)
        return distance
