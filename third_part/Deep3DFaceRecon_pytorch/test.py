"""This script is the test script for Deep3DFaceRecon_pytorch
Change most absolute path to relative path
"""

import os
import sys
from PIL import Image
import numpy as np
import torch

# print(__package__)
__package__ = 'third_part.Deep3DFaceRecon_pytorch'
sys.path.append(os.path.abspath('../../'))
# sys.path.append(os.path.abspath('./models/'))
# sys.path.append(os.path.abspath('./options/'))
# sys.path.append(os.path.abspath('./util/'))
# print(sys.path)
# sys.path.append('.')
# sys.path.append()

from .options.test_options import TestOptions
from .models import create_model
from .util.visualizer import MyVisualizer
from .util.preprocess import align_img
from .util.load_mats import load_lm3d

# from data.flist_dataset import default_flist_reader
# from scipy.io import loadmat, savemat


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            print('no landmark file')
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            # model.test()
            model.forward()
        pred_coeff = {key: model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        import pdb; pdb.set_trace()
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1],
        #     save_results=True, count=i, name=img_name, add_image=False)
        #
        # model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        # model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients


if __name__ == '__main__':
    cmd = '--checkpoints_dir /apdcephfs/share_1290939/feiiyin/TH/PIRender/Deep3DFaceRecon_pytorch/checkpoints \
       --bfm_folder /apdcephfs/share_1290939/feiiyin/TH/PIRender/Deep3DFaceRecon_pytorch/BFM --name=model_name \
       --epoch=20  --img_folder=/apdcephfs/share_1290939/feiiyin/TH/PIRender_bak/Deep3DFaceRecon_pytorch/datasets/examples'
       # --epoch=20  --img_folder=temp'

    opt = TestOptions(cmd_line=cmd).parse()  # get test options
    main(0, opt, opt.img_folder)
    

# python test.py