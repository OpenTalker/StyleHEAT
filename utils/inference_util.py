import glob
import os
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from data.audio_dataset import AudioDataset
from data.inference_dataset import TempVideoDataset, ImageDataset
from pathlib import Path
from utils.video_preprocess.extract_3dmm import Extract3dmm
from utils.common import tensor2img


# intuitive editings
def get_intuitive_control():
    control_dict = {}
    control_dict['rotation_center'] = torch.tensor([0, 0, 0, 0, 0, 0.45])
    control_dict['rotation_left_x'] = torch.tensor([0, 0, math.pi / 10, 0, 0, 0.45])
    control_dict['rotation_right_x'] = torch.tensor([0, 0, -math.pi / 10, 0, 0, 0.45])

    control_dict['rotation_left_y'] = torch.tensor([math.pi / 10, 0, 0, 0, 0, 0.45])
    control_dict['rotation_right_y'] = torch.tensor([-math.pi / 10, 0, 0, 0, 0, 0.45])

    control_dict['rotation_left_z'] = torch.tensor([0, math.pi / 8, 0, 0, 0, 0.45])
    control_dict['rotation_right_z'] = torch.tensor([0, -math.pi / 8, 0, 0, 0, 0.45])

    expression = torch.load('./docs/demo/intuitive_edit/expression.pth')
    for item in ['expression_center', 'expression_mouth', 'expression_eyebrow', 'expression_eyes']:
        control_dict[item] = expression[item]
    # expression = loadmat('/apdcephfs/share_1290939/feiiyin/TH/PIRender_code/comparison_experiment/intuitive_control/expression.mat')
    # for item in ['expression_center', 'expression_mouth', 'expression_eyebrow', 'expression_eyes']:
    #     control_dict[item] = torch.tensor(expression[item])[0]
    # eye = '/apdcephfs/share_1290939/feiiyin/TH/PIRender_code/comparison_experiment/intuitive_control/eye1.npy'
    # eye = torch.tensor(np.load(eye))[0]
    # control_dict['expression_eyes'] = eye

    sort_rot_control = [
        'rotation_left_x', 'rotation_center',
        'rotation_right_x', 'rotation_center',
        'rotation_left_y', 'rotation_center',
        'rotation_right_y', 'rotation_center',
        'rotation_left_z', 'rotation_center',
        'rotation_right_z', 'rotation_center'
    ]

    sort_exp_control = [
        'expression_center', 'expression_mouth',
        'expression_center', 'expression_eyebrow',
        'expression_center', 'expression_eyes',
        'expression_center',
    ]
    return control_dict, sort_rot_control, sort_exp_control


def hfgi_inversion(generator, source_image, args, batch_size=1, inv_path=None):
    assert args.inversion_option in ['load', 'optimize', 'encode']
    if args.attribute_edit:
        args.inversion_option = 'encode'
    if args.inversion_option == 'load':
        assert inv_path is not None, f'inv_path is None.'
        ix, wx, fx = generator.generator.load_FS_results(inv_path)
        inv_data = ix.expand(batch_size, 3, 256, 256), \
                   wx.expand(batch_size, 18, 512), \
                   fx.expand(batch_size, 512, 64, 64), \
                   None
    elif args.inversion_option == 'optimize':
        ix, wx, fx = generator.generator.optimize_inverse(source_image)  # need grad
        inv_data = ix.expand(batch_size, 3, 256, 256), \
                   wx.expand(batch_size, 18, 512), \
                   fx.expand(batch_size, 512, 64, 64), \
                   None
        # tensor2img(ix).save('/apdcephfs/share_1290939/feiiyin/TH/StyleHEAT/docs/demo/output/1.jpg')
        # assert False
    else:
        ix, wx, fx, ada_condition_x = generator.generator.inverse(source_image)
        inv_data = ix.expand(batch_size, 3, 256, 256), \
                   wx.expand(batch_size, 18, 512), \
                   fx.expand(batch_size, 512, 64, 64), \
                   (ada_condition_x[0].expand(batch_size, 512, 64, 64),
                    ada_condition_x[1].expand(batch_size, 512, 64, 64))
    return inv_data

# from models.psp3.pti import RunConfig
# from models.styleheat.styleheat3 import StyleHEAT3
# from models.stylegan3.model import SG3Generator
# def stylegan3_inversion(generator: StyleHEAT3, source_image, args, batch_size=1):
#     if args.inversion_option == 'load':
#         image_name = os.path.basename(args.image_source).split('.')[0]
#         # video_name = os.path.basename(args.video_source)
#         inv_path = os.path.join(args.output_dir, image_name)
#         # default as load pti generator checkpoint and latent
#         assert os.path.exists(inv_path), f'inv_path is None.'
#         ckpt_path = os.path.join(inv_path, f'final_pti_model_{image_name}.pt')
#         generator.generator.decoder = SG3Generator(checkpoint_path=Path(ckpt_path)).decoder
#
#         latent_path = os.path.join(inv_path, f'latents.npy')
#         latent = np.load(latent_path, allow_pickle=True)
#         latent = torch.from_numpy(latent).cuda()
#         ix, wx, fx = generator.generator.decoder.synthesis(
#             latent, return_latents=True, noise_mode='const', force_fp32=True
#         )
#         ix = F.interpolate(ix, (256, 256), mode='bilinear')
#     elif args.inversion_option == 'pti':
#         opt = RunConfig(
#             images_path=Path(args.image_source),
#             latents_path=None,
#             output_path=Path(args.output_dir)
#         )
#         ix, wx, fx = generator.generator.pti_inverse(source_image, opt)  # need grad
#     else:
#         assert False
#         ix, wx, fx = generator.generator.inverse(source_image)
#
#     # Edit pose to make it similar to the target pose via StyleGAN prior
#     # _wx = wx
#     # for factor in range(-20, 20, 1):
#     #     ix, _, fx = generator.generator.edit(_wx, 'pose', factor=factor)
#     #     inv_path = os.path.join(args.output_dir, image_name)
#     #     tensor2img(ix).save(os.path.join(inv_path, f'edit_pose_{factor}.jpg'))
#     # assert False
#
#     # ix, wx, fx = generator.generator.edit(wx, 'pose', factor=-7)
#
#     inv_data = ix.expand(batch_size, 3, 256, 256), \
#                wx.expand(batch_size, 16, 512), \
#                fx.expand(batch_size, 406, 276, 276)
#     return inv_data



def build_history_file_list():
    file_list = [
        'RD_Radio34_003_img_20',
        'RD_Radio34_003_img_40',
        'RD_Radio34_003_img_100',
        'RD_Radio34_003_img_180',
        'RD_Radio34_003_img_280',
        'RD_Radio34_003_img_320',
        'RD_Radio34_003_img_360',
        'RD_Radio34_003_img_420',
        'RD_Radio34_003_img_560',
        'RD_Radio34_003_img_580',
        'RD_Radio34_003_img_600',
        'RD_Radio34_003_img_720',
        'RD_Radio34_003_img_740',
        'RD_Radio34_003_img_760',
        'RD_Radio34_003_img_780',
        'RD_Radio34_003_img_860',
        'RD_Radio34_007_img_281',
        'RD_Radio34_007_img_761',
        'RD_Radio34_007_img_761',
        'RD_Radio34_009_img_382',
        'RD_Radio34_009_img_402',
        'WRA_BobCorker_000_img_4',
        'WRA_BobCorker_000_img_84',
        'WRA_BobCorker_000_img_204',
        'WRA_BobCorker_000_img_264',
        'WRA_BobCorker_000_img_324',
        'WRA_BobCorker_000_img_404',
        'WRA_BobCorker_000_img_444',
        'WRA_BobCorker_000_img_484',
        'WRA_BobCorker_000_img_524',
        'WRA_BobCorker_000_img_544',
        'WRA_BobCorker_000_img_564',
        'WRA_BobCorker_000_img_684',
        'WRA_BobCorker_000_img_764',
        'WRA_BobCorker_000_img_904',
        'WRA_CandiceMiller0_000_img_85',
        'WRA_CandiceMiller0_000_img_285',
        'WRA_CandiceMiller0_000_img_345',
        'WRA_CandiceMiller0_000_img_365',
        'WRA_CandiceMiller0_000_img_385',
        'WRA_CandiceMiller0_000_img_505',
        'WRA_CandiceMiller0_000_img_545',
    ]
    video_list = []
    image_list = []
    video_root = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/sr_video'
    image_root = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/image'
    for i in range(len(file_list)):
        pp = file_list[i].split('_img_')
        video = pp[0] + '.mp4'
        image = pp[1] + '.jpg'
        video_list.append(os.path.join(video_root, video))
        image_list.append(os.path.join(image_root, image))
    print(f'Eg. {video_list[:2]}, {image_list[:2]}')
    return video_list, image_list


def build_inference_dataset(args, opt):
    model_3dmm = None
    if args.if_extract:
        model_3dmm = Extract3dmm()
    start_time = time.time()
    if args.from_dataset:
        dataset = AudioDataset(opt.data, is_inference=True)
    elif args.intuitive_edit:
        assert args.image_source is not None
        if os.path.isdir(args.image_source):
            image_list = sorted(glob.glob(f'{args.image_source}/*.jpg'))
        else:
            image_list = [args.image_source]
        dataset = ImageDataset(image_list, model_3dmm)
    else:
        assert args.video_source is not None

        if os.path.isdir(args.video_source):
            video_list = sorted(glob.glob(f'{args.video_source}/*.mp4'))
        else:
            video_list = [args.video_source]
        # print(video_list)
        if args.cross_id and args.image_source is not None:
            if os.path.isdir(args.image_source):
                image_list = sorted(glob.glob(f'{args.image_source}/*.jpg'))
                # If the names of images need integer sorted
                # image_list = sorted(glob.glob(f'{args.image_source}/*.jpg'),
                #                     key=lambda info: (int(info.split('/')[-1].split('.')[0])))
            else:
                image_list = [args.image_source]
        else:
            image_list = None
        # video_list, image_list = build_history_file_list()
        dataset = TempVideoDataset(
            video_list=video_list, model_3dmm=model_3dmm, if_align=args.if_align,
            cross_id=args.cross_id, image_list=image_list, resize=1024)
    end_time = time.time()
    # print(f'Build dataset (extract 3DMM) time consuming: {end_time - start_time} second.')
    return dataset


# id_list = [
#     600, 40, 100, 120, 380, 840, 940, 181, 261, 281, 541, 601, 661, 941, 322, 342, 602, 642,
#     # 662, 802, 743, 684, 884, 365, 505, 545, 166, 726, 507, 88, 288, 348, 149, 249, 629,
#     # 969, 91, 571, 212, 196, 194, 316, 416, 456, 616, 896, 159, 559, 600
# ]
