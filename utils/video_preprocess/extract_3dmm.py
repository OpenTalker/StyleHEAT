"""This script is the test script for Deep3DFaceRecon_pytorch
Change most absolute path to relative path
"""

import os
import sys
# sys.path.append()
from third_part.Deep3DFaceRecon_pytorch.options.test_options import TestOptions
# from data import create_dataset
from third_part.Deep3DFaceRecon_pytorch.models import create_model
from third_part.Deep3DFaceRecon_pytorch.util.visualizer import MyVisualizer
from third_part.Deep3DFaceRecon_pytorch.util.preprocess import align_img
from PIL import Image
import numpy as np
from third_part.Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
import torch
from configs.path import PRETRAINED_MODELS_PATH


class Extract3dmm:

    def __init__(self):
        bfm_path = PRETRAINED_MODELS_PATH['BFM']
        deep3d_path = PRETRAINED_MODELS_PATH['3DMM']
        deep3d_path = os.path.dirname(deep3d_path)
        deep3d_dir = os.path.dirname(deep3d_path)
        deep3d_name = os.path.basename(deep3d_path)
        cmd = f'--checkpoints_dir {deep3d_dir} ' \
            f'--bfm_folder {bfm_path} --name={deep3d_name} ' \
            f'--epoch=20  --img_folder=temp'
        opt = TestOptions(cmd_line=cmd).parse()  # get test options

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = 'cuda'
        self.model.parallelize()
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder)

    def image_transform(self, images, lm):
        # W, H = images.size
        W, H = 256, 256
        imsize = 256  # Note this hyper-param is key for downloading optical model
        images = images.resize((imsize, imsize))
        # lm = lm * imsize / W  # lm coordinate is corresponding to the image size
        # lm = lm.copy()  # Note that lm has been extracted at the size of 256

        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2] + 1) / 2.
            lm = np.concatenate(
                [lm[:, :1] * W, lm[:, 1:2] * H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)
        img = torch.tensor(np.array(img) / 255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params

    def get_3dmm(self, images_pil, lms):
        """
        :param images: PIL list
        :return:
        """
        images = []
        trans_params = []
        for i, img in enumerate(images_pil):
            lm = lms[i]
            img, lm, p = self.image_transform(img, lm)
            images.append(img)
            trans_params.append(p)

        images = torch.stack(images)
        trans_params = torch.stack(trans_params)

        batch_size = 20
        num_batch = images.shape[0] // batch_size + 1
        pred_coeffs = []
        for _i in range(num_batch):
            _images = images[_i * batch_size: (_i+1) * batch_size]
            if len(_images) == 0:
                break
            data_input = {
                'imgs': _images,
            }
            self.model.set_input(data_input)
            with torch.no_grad():
                self.model.test()
            pred_coeff = {key: self.model.pred_coeffs_dict[key] for key in self.model.pred_coeffs_dict}
            pred_coeff = torch.cat([
                pred_coeff['id'],
                pred_coeff['exp'],
                pred_coeff['tex'],
                pred_coeff['angle'],
                pred_coeff['gamma'],
                pred_coeff['trans']], 1
            )
            _trans_params = np.array(trans_params[_i * batch_size: (_i+1) * batch_size])
            _, _, ratio, t0, t1 = np.hsplit(_trans_params, 5)
            crop_param = np.concatenate([ratio, t0, t1], 1)
            pred_coeff = np.concatenate([pred_coeff.cpu().numpy(), crop_param], 1)

            pred_coeffs.append(pred_coeff)

        coeff_3dmm = np.concatenate(pred_coeffs, 0)

        # extract 73 feature from 260
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:, 80:144]  # expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:, 224:227]  # euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:, 254:257]  # translation
        crop = coeff_3dmm[:, 257:300]  # crop param
        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return coeff_3dmm


