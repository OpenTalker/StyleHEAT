import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class VideoDataset(Dataset):

    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.person_ids = self.person_ids * 100

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids            

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            # print(key)
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data = {}
        person_id = self.person_ids[index]
        # if_same_person = random.randint(0, 1)
        # data['if_same_person'] = if_same_person
        # if if_same_person:
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        num_frame = video_item['num_frame']
        frame_source, frame_target = self.random_select_frames(video_item)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], frame_source)
            img_bytes_1 = txn.get(key)
            key = format_for_lmdb(video_item['video_name'], frame_target)
            img_bytes_2 = txn.get(key)

            key = format_for_lmdb(video_item['video_name'], 'align', frame_source)
            align_bytes_1 = txn.get(key)
            key = format_for_lmdb(video_item['video_name'], 'align', frame_target)
            align_bytes_2 = txn.get(key)

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((num_frame, -1))

            keypoint_key = format_for_lmdb(video_item['video_name'], 'keypoint')
            keypoint_numpy = np.frombuffer(txn.get(keypoint_key), dtype=np.float32)
            keypoint_numpy = keypoint_numpy.reshape((num_frame, 68, 2))

        img1 = Image.open(BytesIO(img_bytes_1))
        data['source_image'] = self.transform(img1)

        img2 = Image.open(BytesIO(img_bytes_2))
        data['target_image'] = self.transform(img2)

        align1 = Image.open(BytesIO(align_bytes_1))
        data['source_align'] = self.transform(align1)

        align2 = Image.open(BytesIO(align_bytes_2))
        data['target_align'] = self.transform(align2)

        # Note: below params are extracted from GT, not align images
        data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source)
        data['target_semantics'] = self.transform_semantic(semantics_numpy, frame_target)

        data['source_keypoint'] = keypoint_numpy[frame_source]
        data['target_keypoint'] = keypoint_numpy[frame_target]
        return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index, ...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:, 80:144]  # expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:, 224:227]  # euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:, 254:257]  # translation
        crop = coeff_3dmm[:, 257:300]  # crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1, 0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [min(max(item, 0), num_frames-1) for item in seq]
        return seq

