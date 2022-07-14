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


class AudioDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        lmdb_path = opt.lmdb_path
        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        audio_lmdb_path = opt.audio_lmdb_path
        self.audio_env = lmdb.open(
            audio_lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        crop_lmdb_path = opt.crop_lmdb_path
        self.crop_env = lmdb.open(
            crop_lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.audio_env:
            raise IOError('Cannot open lmdb dataset', audio_lmdb_path)

        # Note several videos do not has audio files, so changed
        list_file = "audio_test_list.txt" if is_inference else "audio_train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')

        if not is_inference:
            self.person_ids = self.person_ids * 200

        self.transform = transforms.Compose([
            transforms.Resize((opt.resolution, opt.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        # for video load use
        self.video_index = -1
        self.syncnet_mel_step_size = 32

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

        frame_target = max(frame_target, 14)
        frame_target = min(frame_target, num_frame - 14)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], frame_source)
            img_bytes_1 = txn.get(key)
            source_image = Image.open(BytesIO(img_bytes_1))
            # key = format_for_lmdb(video_item['video_name'], frame_target)
            # img_bytes_2 = txn.get(key)

            key = format_for_lmdb(video_item['video_name'], 'align', frame_source)
            align_bytes_1 = txn.get(key)
            source_align = Image.open(BytesIO(align_bytes_1))
            # key = format_for_lmdb(video_item['video_name'], 'align', frame_target)
            # align_bytes_2 = txn.get(key)

            target_img = []
            for _i in range(frame_target - 2, frame_target + 3):
                key = format_for_lmdb(video_item['video_name'], _i)
                img_bytes_2 = txn.get(key)
                img2 = Image.open(BytesIO(img_bytes_2))
                target_img.append(self.transform(img2))

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((num_frame, -1))

            keypoint_key = format_for_lmdb(video_item['video_name'], 'keypoint')
            keypoint_numpy = np.frombuffer(txn.get(keypoint_key), dtype=np.float32)
            keypoint_numpy = keypoint_numpy.reshape((num_frame, 68, 2))

        with self.audio_env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'])
            audio_byte = txn.get(key)
            audio_numpy = np.frombuffer(audio_byte, dtype=np.float64)
            audio_numpy = audio_numpy.reshape(-1, 80)

        with self.crop_env.begin(write=False) as txn:
            crop_key = format_for_lmdb(video_item['video_name'], 'crop_params')
            crop_numpy = np.frombuffer(txn.get(crop_key), dtype=np.int64)
            crop_numpy = crop_numpy.reshape((num_frame, 4))

        target_audio = []
        target_semantics = []
        target_keypoint = []
        target_crop = []
        for _i in range(frame_target - 2, frame_target + 3):
            target_audio.append(self.transform_audio(audio_numpy, _i, syncnet_mel_step_size=self.syncnet_mel_step_size))
            target_semantics.append(self.transform_semantic(semantics_numpy, _i))
            target_keypoint.append(keypoint_numpy[_i])
            target_crop.append(crop_numpy[_i])
        target_audio_sync = self.transform_audio(audio_numpy, frame_target, syncnet_mel_step_size=16)

        data['target_audio'] = torch.stack(target_audio, dim=0)  # 5, 80, 80
        data['target_semantics'] = torch.stack(target_semantics, dim=0)  # 5, 73, 27
        data['target_keypoint'] = torch.from_numpy(np.stack(target_keypoint, axis=0))  # 5, 68, 2
        data['target_crop'] = torch.from_numpy(np.stack(target_crop, axis=0))  # 5, 4
        data['target_image'] = torch.stack(target_img, dim=0)  # 5, 3, 256, 256
        data['target_audio_sync'] = target_audio_sync.unsqueeze(0)  # 1, 80, 16

        data['source_image'] = self.transform(source_image)  # 3, 256, 256
        data['source_align'] = self.transform(source_align)  # 3, 256, 256
        data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source)  # 73, 27

        # for key in data:
        #     print(f'{key}: {data[key].shape}')
        return data

    def load_specific_video(self, audio_name, video_name, source_image_path, source_3dmm, source_image_name=None):
        data = {}
        data['audio_name'] = audio_name
        data['video_name'] = video_name
        if source_image_name is not None:
            data['name'] = f'{audio_name}_{source_image_name}'
        else:
            temp_name = source_image_path.split('/')[-1].split('.')[0]
            data['name'] = f'{audio_name}_{temp_name}'

        transform = transforms.Compose([
            # transforms.Resize((512, 512)),  # size can change adapt for 512
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        frame_start = 15
        frame_end = 265
        # frame_start = 425
        # frame_end = 265

        if source_image_name is not None:
            data['source_image_name'] = source_image_name
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(source_image_name, 'align', 0)
                image = Image.open(BytesIO(txn.get(key)))
                data['source_align'] = transform(image)

                semantics_key = format_for_lmdb(source_image_name, 'coeff_3dmm')

                semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
                semantics_numpy = semantics_numpy.reshape((-1, 260))
                # source_3dmm = torch.from_numpy(semantics_numpy[0]).unsqueeze(0)
                data['source_semantics'] = self.transform_semantic(semantics_numpy, 0)
        else:
            source_align = Image.open(source_image_path)
            data['source_align'] = transform(source_align)
            data['source_image'] = data['source_align']
            data['source_semantics'] = source_3dmm.unsqueeze(-1).repeat(1, 1, 27)

        with self.audio_env.begin(write=False) as txn:
            key = format_for_lmdb(audio_name)
            audio_byte = txn.get(key)
            audio_numpy = np.frombuffer(audio_byte, dtype=np.float64)
            audio_numpy = audio_numpy.reshape(-1, 80)

        print('audio:', audio_name)
        data['audio_image'], data['target_audio'] = [], []
        with self.env.begin(write=False) as txn:
            for frame_index in range(frame_start, frame_end):
                key = format_for_lmdb(audio_name, frame_index)
                image = Image.open(BytesIO(txn.get(key)))
                data['audio_image'].append(transform(image))
                data['target_audio'].append(
                    self.transform_audio(audio_numpy, frame_index, syncnet_mel_step_size=self.syncnet_mel_step_size))
        print('video:', video_name)
        data['video_image'], data['target_semantics'] = [], []
        with self.env.begin(write=False) as txn:
            semantics_key = format_for_lmdb(video_name, 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((-1, 260))

            video_length = len(semantics_numpy)
            # for frame_index in range(video_length - 250, video_length):
            for frame_index in range(frame_start, frame_end):
                key = format_for_lmdb(video_name, frame_index)
                image = Image.open(BytesIO(txn.get(key)))
                data['video_image'].append(transform(image))
                data['target_semantics'].append(self.transform_semantic(semantics_numpy, frame_index))
        # for _i in range(len(self.video_items)):
        #     if self.video_items[_i]['video_name'] == audio_name:

        #         video_item = self.video_items[_i]
        #
        #
        #     if self.video_items[_i]['video_name'] == video_name:
        #
        return data

    def load_next_video(self, image_path=None, video_name=None, if_audio=True):
        data = {}
        self.video_index += 1
        if video_name is not None:
            for _i, item in enumerate(self.video_items):
                if item['video_name'] == video_name:
                    self.video_index = _i
                    video_item = self.video_items[self.video_index]
                    break
        else:
            video_item = self.video_items[self.video_index]
        data['video_name'] = video_item['video_name']

        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # size can change adapt for 512
            # transforms.Resize((256, 256)),  # size can change adapt for 512
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        if if_audio:
            with self.audio_env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'])
                audio_byte = txn.get(key)
                audio_numpy = np.frombuffer(audio_byte, dtype=np.float64)
                audio_numpy = audio_numpy.reshape(-1, 80)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 0)
            source_image = Image.open(BytesIO(txn.get(key)))
            data['source_image'] = transform(source_image)

            key = format_for_lmdb(video_item['video_name'], 'align', 0)
            source_align = Image.open(BytesIO(txn.get(key)))
            data['source_align'] = transform(source_align)

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'], -1))
            data['source_semantics'] = self.transform_semantic(semantics_numpy, 0)

            data['target_image'], data['target_semantics'], data['target_audio'] = [], [], []
            # num_frame = min(video_item['num_frame'], 100)  # change back later TODO: for cross-id use
            num_frame = video_item['num_frame']
            # due to the audio is extracted from an interval, we skip the first half-second frames
            frame_start = 13
            frame_end = num_frame - frame_start
            for frame_index in range(frame_start, frame_end):
                key = format_for_lmdb(video_item['video_name'], frame_index)
                image = Image.open(BytesIO(txn.get(key)))
                data['target_image'].append(transform(image))
                data['target_semantics'].append(self.transform_semantic(semantics_numpy, frame_index))
                if if_audio:
                    data['target_audio'].append(self.transform_audio(audio_numpy, frame_index,
                                                                     syncnet_mel_step_size=self.syncnet_mel_step_size))

        if image_path is not None:
            image = Image.open(image_path).resize((256, 256))
            data['source_align'] = transform(image)
            data['source_image'] = transform(image)
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

    def find_crop_norm_ratio(self, source_coeff, target_coeffs):
        # TODO, used it when reenactment on cross id
        alpha = 0.3
        exp_diff = np.mean(np.abs(target_coeffs[:, 80:144] - source_coeff[:, 80:144]), 1)
        angle_diff = np.mean(np.abs(target_coeffs[:, 224:227] - source_coeff[:, 224:227]), 1)
        index = np.argmin(alpha * exp_diff + (1 - alpha) * angle_diff)
        crop_norm_ratio = source_coeff[:, -3] / target_coeffs[index:index + 1, -3]
        return crop_norm_ratio

    def transform_audio(self, audio_numpy, frame_index, syncnet_mel_step_size):
        fps = 25.
        # syncnet_mel_step_size = 80
        if syncnet_mel_step_size == 80:  # 1s
            frame_index = max(0, frame_index - 12)
            start_idx = int(80. * frame_index / fps)
            end_idx = start_idx + syncnet_mel_step_size
            # print(audio_numpy.shape, start_idx, end_idx)
            out = audio_numpy[start_idx:end_idx, :].transpose(1, 0)
            out = torch.from_numpy(out)
        elif syncnet_mel_step_size == 48:  # 0.6s
            frame_index = max(0, frame_index - 7)
            start_idx = int(80. * frame_index / fps)
            end_idx = start_idx + syncnet_mel_step_size
            # print(audio_numpy.shape, start_idx, end_idx)
            out = audio_numpy[start_idx:end_idx, :].transpose(1, 0)
            out = torch.from_numpy(out)
        elif syncnet_mel_step_size == 32:  # 0.4s
            frame_index = max(0, frame_index - 4)
            start_idx = int(80. * frame_index / fps)
            end_idx = start_idx + syncnet_mel_step_size
            # print(audio_numpy.shape, start_idx, end_idx)
            out = audio_numpy[start_idx:end_idx, :].transpose(1, 0)
            out = torch.from_numpy(out.copy())
        elif syncnet_mel_step_size == 16:  # 0.2s
            frame_index = max(0, frame_index - 2)
            start_idx = int(80. * frame_index / fps)
            end_idx = start_idx + syncnet_mel_step_size
            # start_idx = 0
            # end_idx = 16
            # print(audio_numpy.shape, start_idx, end_idx)
            out = audio_numpy[start_idx:end_idx, :].transpose(1, 0)
            out = torch.from_numpy(out)
        else:
            raise NotImplementedError
        return out

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index - self.semantic_radius, index + self.semantic_radius + 1))
        seq = [min(max(item, 0), num_frames - 1) for item in seq]
        return seq

