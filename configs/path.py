from pathlib import Path

PRETRAINED_MODELS_PATH = {
    # models
    'e4e': 'checkpoints/Encoder_e4e.pth',
    'hfgi': 'checkpoints/hfgi.pth',
    'stylegan2': 'checkpoints/StyleGAN_e4e.pth',
    # editing
    'interfacegan': 'checkpoints/interfacegan_directions/',
    'ganspace': 'checkpoints/ffhq_pca.pt',
    'FFHQ_PCA': 'checkpoints/ffhq_PCA.npz',
    '': '',
    # pretrain
    'discriminator': 'checkpoints/stylegan2_d_256.pth',
    'video_warper': 'checkpoints/video_warper.pth',
    'styleheat': 'checkpoints/StyleHEAT_visual.pt',
    # id_loss
    'irse50': 'checkpoints/model_ir_se50.pth',
    # 3DMM
    'BFM': 'checkpoints/BFM',
    '3DMM': 'checkpoints/Deep3D/epoch_20.pth',
}
