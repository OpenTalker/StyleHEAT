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

        gt_images.append(target_images)
        fake_images.append(output['fake_image'].cpu().clamp_(-1, 1))
        video_warp_images.append(output['video_warp_image'].cpu().clamp_(-1, 1))

    fake_images = torch.cat(fake_images, 0)
    # gt_images = torch.cat(gt_images, 0)
    # video_warp_images = torch.cat(video_warp_images, 0)

    video_util.write2video("{}/{}".format(args.output_dir, data['video_name']), fake_images)
    print('Save video in {}/{}.mp4'.format(args.output_dir, data['video_name']))


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

