import argparse
import tqdm
import torch
import os
import utils.inference_util as inference_util
import utils.video_util as video_util
from utils.common import tensor2img
from configs.config import Config
from configs.path import PRETRAINED_MODELS_PATH
from models.styleheat.styleheat import StyleHEAT


def reenactment(generator, data):
    """
    :param generator:
    :param data: {
        video_name: <class 'str'>
        source_image: <class 'torch.Tensor'> (3, 256, 256)
        source_semantics: <class 'torch.Tensor'> (73, 27)
        target_image: <class 'list'> (B, 3, 256, 256)
        target_semantics: <class 'list'> (B, 73, 27)
        target_audio: <class 'list'> (B, 80, 16)
    }
    :return:
    """
    bs = args.batch_size
    source_image = data['source_align'].unsqueeze(0).cuda()
    inv_data = inference_util.hfgi_inversion(generator, source_image, args=args, batch_size=bs)
    source_image = source_image.repeat(bs, 1, 1, 1)
    
    num_batch = len(data['target_semantics']) // bs + 1
    gt_images, video_warp_images, audio_warp_images, fake_images = [], [], [], []
    source_3dmm = data['source_semantics'].unsqueeze(-1).repeat(1, 1, 27)  # 1, 73, 27
    for _i in range(num_batch):
        target_3dmm = data['target_semantics'][_i * bs:(_i + 1) * bs]
        if len(target_3dmm) == 0 or _i * bs > args.frame_limit:
            break
        
        if not torch.is_tensor(target_3dmm):
            target_3dmm = torch.stack(target_3dmm).cuda()
        else:
            target_3dmm = target_3dmm.cuda()

        if 'target_image' in data:
            target_images = data['target_image'][_i * bs:(_i + 1) * bs]
        else:
            target_images = None

        _len_3dmm = len(target_3dmm)
        if _len_3dmm < bs:
            # Last batch
            ix, wx, fx, inversion_condition = inv_data
            ix, wx, fx = ix[:_len_3dmm], wx[:_len_3dmm], fx[:_len_3dmm]
            if args.inversion_option == 'encode':
                inversion_condition = (inversion_condition[0][:_len_3dmm], inversion_condition[1][:_len_3dmm])
            inv_data = ix, wx, fx, inversion_condition

            source_3dmm = source_3dmm[:_len_3dmm]

        with torch.no_grad():
            if args.edit_expression_only:
                target_3dmm[:, 64:, :] = source_3dmm[:, 64:, :]
            output = generator.forward(source_image, target_3dmm, inv_data=inv_data, imsize=1024)

        # gt_images.append(target_images)
        fake_images.append(output['fake_image'].cpu().clamp_(-1, 1))
        # video_warp_images.append(output['video_warp_image'].cpu().clamp_(-1, 1))

    fake_images = torch.cat(fake_images, 0)
    # gt_images = torch.cat(gt_images, 0)
    # video_warp_images = torch.cat(video_warp_images, 0)

    video_util.write2video("{}/{}".format(args.output_dir, data['video_name']), fake_images)
    print('Save video in {}/{}.mp4'.format(args.output_dir, data['video_name']))


def audio_reenactment(generator, data, audio_path):
    # import TODO
    import sys
    import os
    sys.path.append(os.path.abspath('./third_part/SadTalker/'))
    import random
    import numpy as np
    from pydub import AudioSegment 
    from third_part.SadTalker.src.test_audio2coeff import Audio2Coeff
    from third_part.SadTalker.src.generate_batch import get_data
    from third_part.SadTalker.src.utils.preprocess import CropAndExtract
    import scipy.io as scio
    import warnings
    warnings.filterwarnings("ignore")

    current_root_path = os.path.abspath('./third_part/SadTalker/')
    checkpoint_dir = './checkpoints/'
    device = 'cuda'
    temp_root = './docs/demo/output/temp/'
    os.makedirs(temp_root, exist_ok=True)

    wav2lip_checkpoint = os.path.join(current_root_path, checkpoint_dir, 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, checkpoint_dir, 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')
    
    audio2exp_checkpoint = os.path.join(current_root_path, checkpoint_dir, 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, checkpoint_dir, 'facevid2vid_00189-model.pth.tar')
    mapping_checkpoint = os.path.join(current_root_path, checkpoint_dir, 'mapping_00229-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, 
                                audio2exp_checkpoint, audio2exp_yaml_path, 
                                wav2lip_checkpoint, device)

    first_coeff_path = os.path.join(temp_root, 'first_coeff.npy')
    source_3dmm = data['source_semantics'][0, :-3].unsqueeze(0) # Reduce the crop params
    crop_3dmm = data['source_semantics'][0, -3:].unsqueeze(0)
    source_3dmm = source_3dmm.cpu().numpy()
    # print(source_3dmm.shape)
    # import pdb; pdb.set_trace()
    scio.savemat(first_coeff_path, {'coeff_3dmm': source_3dmm})

    '''
    # path_of_lm_croper = os.path.join(current_root_path, checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    # path_of_net_recon_model = os.path.join(current_root_path, checkpoint_dir, 'epoch_20.pth')
    # dir_of_BFM_fitting = os.path.join(current_root_path, checkpoint_dir, 'BFM_Fitting')
    # preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)
    # first_frame_dir = os.path.join(temp_root, 'first_frame_dir')
    # os.makedirs(first_frame_dir, exist_ok=True)
    
    # pic_path = os.path.join(first_frame_dir, 'first.jpg')
    # tensor2img(data['source_align']).save(pic_path)
    
    # first_coeff_path, crop_pic_path =  preprocess_model.generate(pic_path, first_frame_dir)
    '''

    audio_batch = get_data(first_coeff_path, audio_path, device)
    pose_style = random.randint(0, 45)
    coeff_path = audio_to_coeff.generate(audio_batch, temp_root, pose_style)

    audio_coeff = scio.loadmat(coeff_path)
    audio_coeff = torch.from_numpy(audio_coeff['coeff_3dmm'])
    audio_coeff = torch.cat([audio_coeff, crop_3dmm.repeat(audio_coeff.shape[0], 1)], dim=1)
    
    # data['target_semantics'][0][:, 0][-3:]
    # audio_coeff[0][-3:]
    # import pdb; pdb.set_trace()
    # print('Audio coeff shape: ', audio_coeff.shape)
    semantic_radius = 13
    def obtain_seq_index(index, num_frames):
        seq = list(range(index - semantic_radius, index + semantic_radius + 1))
        seq = [min(max(item, 0), num_frames - 1) for item in seq]
        return seq

    def transform_semantic(semantic, frame_index):
        index = obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index, ...]
        return torch.Tensor(coeff_3dmm).permute(1, 0)

    audio_coeff_list = []
    for _i in range(len(audio_coeff)):
        audio_coeff_list.append(transform_semantic(audio_coeff, _i))
    audio_coeff = torch.stack(audio_coeff_list)
    
    data['target_semantics'] = audio_coeff.to(device)
    
    reenactment(generator, data)
    # import pdb; pdb.set_trace()
    video_path = os.path.join(args.output_dir, data['image_name'] + '.mp4')

    audio_path = audio_path
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
    new_audio_path = os.path.join(args.output_dir, audio_name+'.wav')
    start_time = 0
    sound = AudioSegment.from_mp3(audio_path)

    frames = len(data['target_semantics'])
    end_time = start_time + frames*1/25*1000
    word1 = sound.set_frame_rate(16000)
    word = word1[start_time:end_time]
    word.export(new_audio_path, format="wav")

    av_path = os.path.join(args.output_dir, data['image_name'] + '_audio.mp4')
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, new_audio_path, av_path)
    os.system(cmd)


def attribute_edit(generator, data):
    assert args.attribute in ['young', 'old', 'beard', 'lip']
    # Recommend factor
    if args.attribute == 'young':
        factor = -5.0
    elif args.attribute == 'old':
        factor = 5.0
    elif args.attribute == 'beard':
        factor = -20.0
    elif args.attribute == 'lip':
        factor = 20.0

    bs = args.batch_size
    source_image = data['source_align'].unsqueeze(0).cuda()
    per_wx, per_ix, per_res = generator.generator.encode(source_image)

    inv_data = inference_util.hfgi_inversion(generator, source_image, args=args, batch_size=bs)
    source_image = source_image.repeat(bs, 1, 1, 1)

    num_batch = len(data['target_image']) // bs + 1
    gt_images, video_warp_images, audio_warp_images, fake_images = [], [], [], []
    source_3dmm = data['source_semantics'].unsqueeze(-1).repeat(1, 1, 27)  # 1, 73, 27
    for _i in range(num_batch):
        target_images = data['target_image'][_i * bs:(_i + 1) * bs]
        if len(target_images) == 0 or _i * bs > args.frame_limit:
            break
        target_3dmm = data['target_semantics'][_i * bs:(_i + 1) * bs]
        target_3dmm = torch.stack(target_3dmm).cuda()

        _len_3dmm = len(target_3dmm)
        if _len_3dmm < bs:
            # Last batch
            ix, wx, fx, inversion_condition = inv_data
            ix, wx, fx = ix[:_len_3dmm], wx[:_len_3dmm], fx[:_len_3dmm]
            if args.inversion_option == 'encode':
                inversion_condition = (inversion_condition[0][:_len_3dmm], inversion_condition[1][:_len_3dmm])
            inv_data = ix, wx, fx, inversion_condition
            source_3dmm = source_3dmm[:_len_3dmm]

        with torch.no_grad():
            if args.edit_expression_only:
                target_3dmm[:, 64:, :] = source_3dmm[:, 64:, :]
            output = generator.forward(source_image, target_3dmm, inv_data=inv_data, imsize=1024)

        ix_edit, wx_edit, fx_edit, inversion_condition = generator. \
            generator.edit(x=None, factor=factor / num_batch * (_i + 1), choice=args.attribute, wx=per_wx, res=per_res)
        inv_data = [
            ix_edit.expand(bs, 3, 256, 256),
            wx_edit.expand(bs, 18, 512),
            fx_edit.expand(bs, 512, 64, 64),
            (inversion_condition[0].expand(bs, 512, 64, 64),
             inversion_condition[1].expand(bs, 512, 64, 64))
        ]

        gt_images.append(target_images)
        fake_images.append(output['fake_image'].cpu().clamp_(-1, 1))
        video_warp_images.append(output['video_warp_image'].cpu().clamp_(-1, 1))

    fake_images = torch.cat(fake_images, 0)
    gt_images = torch.cat(gt_images, 0)
    video_warp_images = torch.cat(video_warp_images, 0)

    video_util.write2video("{}/{}".format(args.output_dir, data['video_name']) + '_attribute_edit', fake_images)


def intuitive_edit(generator, data):
    source_image = data['source_align'].unsqueeze(0).cuda()
    inv_data = inference_util.hfgi_inversion(generator, source_image, args=args, batch_size=1)

    control_dict, sort_rot_control, sort_exp_control = inference_util.get_intuitive_control()
    step = 10
    output_images = []
    # rotation control
    current = control_dict['rotation_center']
    data['source_semantics'] = data['source_semantics'].unsqueeze(-1).repeat(1, 1, 27)
    for control in sort_rot_control:
        rotation = None
        for i in range(step):
            rotation = (control_dict[control] - current) * i / (step - 1) + current
            data['source_semantics'][:, 64:70, :] = rotation[None, :, None]
            with torch.no_grad():
                output = generator.forward(source_image, data['source_semantics'].cuda(), inv_data=inv_data, imsize=1024)
            output_images.append(output['fake_image'].cpu().clamp_(-1, 1))
        current = rotation

    # expression control
    current = data['source_semantics'][0, :64, 0]
    for control in sort_exp_control:
        expression = None
        for i in range(step):
            expression = (control_dict[control] - current) * i / (step - 1) + current
            data['source_semantics'][:, :64, :] = expression[None, :, None]
            with torch.no_grad():
                output = generator.forward(source_image, data['source_semantics'].cuda(), inv_data=inv_data, imsize=1024)
            output_images.append(output['fake_image'].cpu().clamp_(-1, 1))
        current = expression
    output_images = torch.cat(output_images, 0)

    video_util.write2video("{}/{}".format(args.output_dir, data['image_name']) + '_intuitive_edit', output_images)


def parse_args():
    parser = argparse.ArgumentParser(description='Inferencing')
    parser.add_argument('--config', default='./configs/inference.yaml')
    parser.add_argument('--name', default='test')
    parser.add_argument('--from_dataset', action='store_true')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--image_source', type=str, default=None, help='Single path or directory')
    parser.add_argument('--video_source', type=str, default=None, help='Single path or directory')
    parser.add_argument('--output_dir', default='./')
    parser.add_argument('--inversion_option', type=str, default='encode', help='load, optimize, encode')
    parser.add_argument('--frame_limit', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--if_extract', action='store_true')
    parser.add_argument('--if_align', action='store_true')
    # Audio
    parser.add_argument('--enable_audio', action='store_true')
    parser.add_argument('--audio_path', type=str, default=None)
    # Editing
    parser.add_argument('--attribute_edit', action='store_true')
    parser.add_argument('--attribute', type=str, default=None)
    parser.add_argument('--intuitive_edit', action='store_true')
    parser.add_argument('--edit_expression_only', action='store_true')
    args = parser.parse_args()
    return args


def main():
    opt = Config(args.config)
    opt.model.enable_audio = args.enable_audio
    generator = StyleHEAT(opt.model, PRETRAINED_MODELS_PATH).cuda()
    dataset = inference_util.build_inference_dataset(args, opt)
    
    for _ in tqdm.tqdm(range(len(dataset))):
        data = dataset.load_next_video()
        if args.intuitive_edit:
            intuitive_edit(generator, data)
        elif args.attribute_edit:
            attribute_edit(generator, data)
        elif args.audio_path is not None:
            audio_reenactment(generator, data, args.audio_path)
        else:
            reenactment(generator, data)


if __name__ == '__main__':
    args = parse_args()
    if args.cross_id:
        print('Cross-id testing')
    else:
        print('Same-id testing')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main()

