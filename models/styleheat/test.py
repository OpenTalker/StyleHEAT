import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.abspath('.'))
from os.path import join
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from configs.path import PRETRAINED_MODELS_PATH
from configs.config import Config
from models.styleheat.styleheat import StyleHEAT
from models.styleheat.mirror_warper import MirrorWarper
from utils.common import tensor2img


def test_styleheat():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./configs/inference.yaml')
    parser.add_argument('--name', default='Name')
    args = parser.parse_args()

    example_opt = Config(args.config, args, is_train=True)
    model = StyleHEAT(example_opt.model, PRETRAINED_MODELS_PATH).cuda()

    input_image = torch.randn(2, 3, 256, 256).cuda()
    driven_3dmm = torch.randn(2, 73, 27).cuda()
    driven_audio = torch.randn(2, 1, 80, 16).cuda()
    out = model(input_image, driven_3dmm, driven_audio)
    print(f'Forward test done.')


def test_styleheat3():
    from models.styleheat.styleheat3 import StyleHEAT3
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./configs/video_styleheat3_trainer.yaml')
    parser.add_argument('--name', default='Name')
    args = parser.parse_args()

    example_opt = Config(args.config, args, is_train=True)
    model = StyleHEAT3(example_opt.model, PRETRAINED_MODELS_PATH).cuda()

    input_image = torch.randn(2, 3, 256, 256).cuda()
    driven_3dmm = torch.randn(2, 73, 27).cuda()
    driven_audio = torch.randn(2, 1, 80, 16).cuda()
    out = model(input_image, driven_3dmm, driven_audio)
    print(f'Forward test done.')


def test_video_warper():
    import torch
    p = '/apdcephfs/share_1290939/feiiyin/TH/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
    a = torch.load(p)['net_G_ema']
    from models.styleheat.warper import VideoWarper
    model = VideoWarper()
    t = {}
    for k in a:
        if k.startswith('warpping_net'):
            t['warping_net' + k[12:]] = a[k]
        # if k.startswith('warping_net'):
        #     t['warping_net' + k[12:]] = a['net_G_ema'][k]
        elif k.startswith('mapping_net'):
            t[k] = a[k]
    model.load_state_dict(t)
    t = {'net_G_ema': t}
    torch.save(t, '/apdcephfs/share_1290939/feiiyin/TH/ft_local/video_warper.pth')


def test_flip_flow_field():
    import torch
    p = '/apdcephfs/share_1290939/feiiyin/TH/ft_local/video_warper.pth'
    t = torch.load(p)['net_G_ema']
    from models.styleheat.warper import VideoWarper
    import utils.flow_util as flow_util
    import numpy as np
    model = VideoWarper().cuda()

    model.load_state_dict(t)

    coeff_3dmm_p = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/sr_video/3dmm/3dmm_RD_Radio34_003.npy'
    coeff_3dmm = np.load(coeff_3dmm_p)
    coeff_3dmm = torch.from_numpy(coeff_3dmm).cuda()
    coeff_3dmm = coeff_3dmm[:1, :, None].repeat(1, 1, 27)
    print(coeff_3dmm.shape)

    image = load_test_image()
    out = model(image, coeff_3dmm)
    root = '/apdcephfs/share_1290939/feiiyin/TH/StyleHEAT_result/test/image'
    tensor2img(out['warp_image']).save(os.path.join(root, 'ori.jpg'))

    flow = out['flow_field']
    print(flow.shape)

    mirror = torch.flip(image, dims=[3])

    flow = torch.flip(flow, dims=[3])
    bad_deformation = flow_util.convert_flow_to_deformation(flow)
    bad_mirror_output = flow_util.warp_image(mirror, bad_deformation)
    tensor2img(bad_mirror_output).save(os.path.join(root, 'bad_mirror.jpg'))

    flow[:, 0, :, :] = flow[:, 0, :, :] * -1
    good_deformation = flow_util.convert_flow_to_deformation(flow)
    good_mirror_output = flow_util.warp_image(mirror, good_deformation)
    tensor2img(good_mirror_output).save(os.path.join(root, 'good_mirror.jpg'))



def test_train():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.a = nn.Conv2d(3, 3, 1, 1)
            self.freeze()

        def freeze(self):
            self.a.requires_grad = False

        def forward(self, x):
            return x

    net = Net()
    print(net.a.requires_grad)
    net.train()
    print(net.a.requires_grad)


def test_same():
    pp = '/apdcephfs/share_1290939/feiiyin/TH/result/refine_v512_v51_regular_global_gan_final/epoch_00010_iteration_000023000_checkpoint.pt'
    p = '/apdcephfs/share_1290939/feiiyin/TH/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
    a = torch.load(p)['net_G_ema']
    aa = torch.load(pp)['net_G_ema']
    for k in a:
        if k.startswith('warpping_net'):
            t = a[k]
            tt = aa[k]
            if torch.sum(t - tt) != 0:
                print(k, torch.sum(t - tt))


def load_test_image():
    # path = '/apdcephfs/share_1290939/feiiyin/TH/Barbershop/input/face/RD_Radio10_000.png'
    path = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/image/403.jpg'

    x = Image.open(path)
    loader = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    x = loader(x)
    x = x.unsqueeze(0).cuda()
    return x


def test_mirror_warper():
    image = load_test_image()
    model = MirrorWarper().cuda()
    # image = torch.randn(2, 3, 256, 256)
    image = image.repeat(2, 1, 1, 1)
    descriptor = torch.randn(2, 68, 27).cuda()
    out = model(image, descriptor)
    print(out['final_image'].shape)
    print(out['descriptor'].shape)


def test_image_mirror_warper():
    from models.styleheat.pirender_mirror_warper import MirrorWarper
    image = load_test_image()
    model = MirrorWarper().cuda()
    descriptor = torch.randn(1, 73, 27).cuda()
    out = model(image, descriptor)
    print(out['final_image'].shape)
    # print(out['descriptor'].shape)


def test_reshape():
    a = torch.tensor([1, 2, 3, 4])
    a = a.reshape(2, 2)
    print(a)
    b = torch.randn(2, 4, 224, 224)
    c = torch.zeros(2, 2, 2, 224, 224)
    c[:, 0] = b[:, :2]
    c[:, 1] = b[:, 2:]
    d = b.reshape(2, 2, 2, 224, 224)
    print(torch.sum(c != d))


def test_extreme_case():
    from models.styleheat.pirender_mirror_warper2 import MirrorWarper
    model = MirrorWarper().cuda()
    path = '/apdcephfs/share_1290939/feiiyin/TH/StyleHEAT_result/train_video_warper_pirender_image_only_mask/epoch_00028_iteration_000036204_checkpoint.pt'
    ckpt = torch.load(path)['net_G_ema']
    model.load_state_dict(ckpt)

    pp = '/apdcephfs/share_1290939/feiiyin/TH/giraffe/out/ffhq256_pretrained/rendering/rotation_object/source/3dmm/3dmm.npy'
    coeff_3dmm = torch.from_numpy(np.load(pp))
    image = load_test_image()
    root = '/apdcephfs/share_1290939/feiiyin/TH/StyleHEAT_result/train_video_warper_pirender_image_only_mask'
    for i in range(len(coeff_3dmm)):
        cc = coeff_3dmm[i]
        cc = cc.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 27).cuda()
        out = model.forward(image, cc)
        tensor2img(out['w_source_image']).save(join(root, f'w_source_image_{i}.jpg'))
        tensor2img(out['w_mirror_image']).save(join(root, f'w_mirror_image_{i}.jpg'))
        tensor2img(out['final_image']).save(join(root, f'final_image_{i}.jpg'))



if __name__ == '__main__':
    # test_styleheat()
    # test_train()
    # test_same()
    # test_styleheat3()
    # test_mirror_warper()
    # test_reshape()
    # test_flip_flow_field()
    # test_image_mirror_warper()
    test_extreme_case()



# CUDA_VISIBLE_DEVICES=0 python models/styleheat/test.py
