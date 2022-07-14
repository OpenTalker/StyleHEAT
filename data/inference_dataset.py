import os
import cv2
from torchvision import transforms
from utils.video_preprocess.align_face import align_image, align_image_pil
from utils.video_preprocess.extract_landmark import get_landmark
from utils.video_preprocess.crop_videos_inference import Croper
import utils.video_util as video_util
import numpy as np
from PIL import Image
import torch


class ImageDataset:

    def __init__(self, image_list, model_3dmm):
        self.model_3dmm = model_3dmm
        self.image_list = image_list
        self.image_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        self.image_index = -1

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image_pil = Image.open(image_path)
        image_name = os.path.basename(image_path).split('.')[0]

        save_3dmm_path = os.path.join(os.path.dirname(image_path), '3dmm', '3dmm_' + image_name + '.npy')
        if not os.path.exists(save_3dmm_path):
            image_pil_256 = image_pil.resize((256, 256))
            os.makedirs(os.path.join(os.path.dirname(image_path), '3dmm'), exist_ok=True)
            # Re-crop

            lm_np = get_landmark([image_pil_256])

            coeff_3dmm = self.model_3dmm.get_3dmm([image_pil_256], lm_np)
            np.save(save_3dmm_path, coeff_3dmm)

        image = self.image_transform(image_pil)
        coeff_3dmm = np.load(save_3dmm_path, allow_pickle=True)
        coeff_3dmm = torch.from_numpy(coeff_3dmm)
        # print(coeff_3dmm.shape) # (1, 73)
        return {
            'source_align': image,
            'source_image': image,
            'image_name': image_name,
            'source_semantics': coeff_3dmm
        }

    def load_next_video(self):
        self.image_index += 1
        return self.__getitem__(self.image_index)


class TempVideoDataset:

    def __init__(self, video_list, model_3dmm, if_align=False, cross_id=False, image_list=None, resize=256):
        self.video_list = video_list
        self.model_3dmm = model_3dmm
        self.cross_id = cross_id
        self.image_list = image_list
        self.if_align = if_align
        if self.cross_id and len(self.video_list) != len(self.image_list):
            self.video_list = self.video_list * (len(self.image_list) // len(self.video_list) + 1)
            self.video_list = self.video_list[:len(self.image_list)]

        self.video_index = -1
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        self.semantic_radius = 13
        self.croper = Croper()

    def __len__(self):
        return len(self.video_list)

    def data_preprocess(self, video_path, image_path=None):
        # Hard code; Bad writing
        video_name = os.path.basename(video_path).split('.')[0]
        frames_pil = video_util.read_video(video_path, resize=256)

        save_3dmm_path = os.path.join(os.path.dirname(video_path), '3dmm', '3dmm_' + video_name + '.npy')
        if not os.path.exists(save_3dmm_path):
            os.makedirs(os.path.join(os.path.dirname(video_path), '3dmm'), exist_ok=True)
            lm_np = get_landmark(frames_pil)

            frames_pil = self.croper.crop(frames_pil, lm_np)
            lm_np = get_landmark(frames_pil)

            coeff_3dmm = self.model_3dmm.get_3dmm(frames_pil, lm_np)
            # print(coeff_3dmm.shape)
            np.save(save_3dmm_path, coeff_3dmm)

        coeff_3dmm = np.load(save_3dmm_path, allow_pickle=True)
        coeff_3dmm = torch.from_numpy(coeff_3dmm)

        if self.cross_id and image_path is not None:
            src_image_pil = Image.open(image_path).convert("RGB")  # prevent png exist channel error

            image_name = os.path.basename(image_path).split('.')[0]
            source_3dmm_path = os.path.join(os.path.dirname(image_path), '3dmm', '3dmm_' + image_name + '.npy')
            if not os.path.exists(source_3dmm_path):
                src_image_pil_256 = src_image_pil.resize((256, 256))
                os.makedirs(os.path.join(os.path.dirname(image_path), '3dmm'), exist_ok=True)
                lm_np = get_landmark([src_image_pil_256])
                # print(lm_np.shape)
                source_3dmm = self.model_3dmm.get_3dmm([src_image_pil_256], lm_np)
                # print(coeff_3dmm.shape)
                np.save(source_3dmm_path, source_3dmm)

            source_3dmm = np.load(source_3dmm_path, allow_pickle=True)
            source_3dmm = torch.from_numpy(source_3dmm)
        else:
            src_image_pil = frames_pil[0]
            source_3dmm = coeff_3dmm[0].unsqueeze(0)

        if self.if_align:
            src_lm_np = get_landmark([src_image_pil])
            src_align_pil = align_image_pil([src_image_pil], src_lm_np)
            src_align_pil = src_align_pil[0]
        else:
            src_align_pil = src_image_pil

        return {
            'source_align': src_align_pil,
            'source_image': src_image_pil,
            'source_3dmm': source_3dmm,
            'frames': frames_pil,
            'coeff_3dmm': coeff_3dmm,
            'video_name': video_name
        }

    def transform_semantic(self, semantic, frame_index):
        # 500, 73
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index, ...]
        # 27, 73
        return torch.Tensor(coeff_3dmm).permute(1, 0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index - self.semantic_radius, index + self.semantic_radius + 1))
        seq = [min(max(item, 0), num_frames - 1) for item in seq]
        return seq

    def find_crop_norm_ratio(self, source_coeff, target_coeffs):
        alpha = 0.3
        # import pdb; pdb.set_trace()
        exp_diff = torch.mean(torch.abs(target_coeffs[:, 0:64] - source_coeff[:, 0:64]), 1)
        angle_diff = torch.mean(torch.abs(target_coeffs[:, 64:67] - source_coeff[:, 64:67]), 1)
        index = torch.argmin(alpha * exp_diff + (1 - alpha) * angle_diff)
        crop_norm_ratio = source_coeff[:, -3] / target_coeffs[index:index + 1, -3]
        return crop_norm_ratio

    def load_next_video(self):
        data = {}
        self.video_index += 1
        video_path = self.video_list[self.video_index]

        if self.cross_id:
            image_path = self.image_list[self.video_index]
            image_name = os.path.basename(image_path).split('.')[0]

            video_data = self.data_preprocess(video_path, image_path)

            data['image_name'] = image_name
            data['video_name'] = f'{video_data["video_name"]}_{image_name}'
        else:
            video_data = self.data_preprocess(video_path)
            data['video_name'] = video_data['video_name']

        data['source_image'] = self.image_transform(video_data['source_image'])
        data['source_align'] = self.image_transform(video_data['source_align'])
        data['source_semantics'] = video_data['source_3dmm']

        frames_pil = video_data['frames']
        frames = [self.transform(i) for i in frames_pil]
        frames = torch.stack(frames)

        data['target_image'] = frames
        data['target_semantics'] = []
        semantics_numpy = video_data['coeff_3dmm']

        for frame_index in range(len(frames)):
            data['target_semantics'].append(self.transform_semantic(semantics_numpy, frame_index))
        return data

