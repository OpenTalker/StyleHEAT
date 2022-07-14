import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
# from utils.distributed import master_only_print as print


class SyncLoss(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.syncnet_T = 5
        self.syncnet_mel_step_size = 16

        self.syncnet = SyncNet()
        path = '/apdcephfs/share_1290939/feiiyin/TH/result/audio_sync_net/checkpoints/checkpoint_step000063000.pth'
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        self.syncnet.load_state_dict(ckpt)
        for p in self.syncnet.parameters():
            p.requires_grad = False
        self.syncnet = self.syncnet.to(device)
        self.syncnet.eval()

        self.logloss = nn.BCELoss()
        print('SyncLoss init done.')

    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.logloss(d.unsqueeze(1), y)
        return loss

    def crop(self, img, crop_param):
        # crop_imgs = []
        # for _i in range(len(img)):
        #     x1, y1, x2, y2 = crop_param[_i]
        #     crop_imgs.append(img[_i, :, y1:y2, x1:x2])
        # crop_imgs = torch.cat(crop_imgs)
        _i = torch.arange(img.shape[0]).unsqueeze(1).to(self.device)
        bbox = torch.cat([_i, crop_param], dim=1).float().to(self.device)
        crop_imgs = roi_align(img, boxes=bbox, output_size=96)

        bs = img.shape[0] // 5
        crop_imgs = crop_imgs.reshape(bs, 5, 3, 96, 96).transpose(1, 2)
        return crop_imgs

    def forward(self, target_audio, fake_image, target_crop):
        r"""Perceptual loss forward.

        Args:
           target_audio (4D tensor) : B * 5, 1, 80, 16
           fake_image: B * 5, 3, 256, 256
           target_crop: B * 5, 4

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        mel = target_audio  # [2::5]
        g = self.crop(fake_image, target_crop)
        # visualize
        # print(f'[LOG]: mel: {mel.shape}, g: {g.shape}')
        # mel = torch.randn(2, 1, 80, 16).cuda()
        # g = torch.randn(2, 3, 5, 96, 96).cuda()

        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(self.syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(self.device)
        return self.cosine_loss(a, v, y)


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):
    
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences):  # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        return audio_embedding, face_embedding


def test_sync_loss():
    batch = 2
    target_audio = torch.randn(batch * 5, 80, 16).cuda()
    fake_image = torch.randn(batch * 5, 3, 256, 256).cuda()
    target_crop = torch.randn(batch * 5, 4).long().cuda()

    loss = SyncLoss()
    output = loss(target_audio, fake_image, target_crop)
    print(output)
