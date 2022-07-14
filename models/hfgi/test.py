import os
import sys
sys.path.append(os.path.abspath('.'))
from PIL import Image
from torchvision import transforms

from configs.path import PRETRAINED_MODELS_PATH
from models.hfgi.hfgi import HFGI
from utils.common import tensor2img


def test_inversion(model, image):
    ix, wx, fx, inversion_condition = model.inverse(image)
    tensor2img(ix).save('inversion.jpg')
    tensor2img(image).save('input.jpg')


def test_editing(model, image):
    # ix_edit, wx_edit, fx_edit, inversion_condition = model.edit(image, factor=4, choice='age')
    root = '/apdcephfs/share_1290939/feiiyin/TH/StyleHEAT_result/interfacegan'
    os.makedirs(root, exist_ok=True)
    for d in range(30):
        degree = (d - 15) / 5.0
        ix_edit, wx_edit, fx_edit, inversion_condition = model.edit(image, factor=degree, choice='pose', output_size=1024)
        tensor2img(ix_edit).save(os.path.join(root, f'editing_{degree}.jpg'))
        tensor2img(image).save(os.path.join(root, f'input.jpg'))


def test_optimize_inverse(model, image):
    ix, w_latent, f_latent = model.optimize_inverse(image, save_path='./')
    tensor2img(image).save('input.jpg')


def load_test_image():
    # path = '/apdcephfs/share_1290939/feiiyin/TH/Barbershop/input/face/RD_Radio10_000.png'
    path = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/image/1.jpg'

    x = Image.open(path)
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    x = loader(x)
    x = x.unsqueeze(0).cuda()
    return x


def main():
    model = HFGI().cuda()
    model.load_checkpoint(PRETRAINED_MODELS_PATH)

    x = load_test_image()
    # test_inversion(model, x)
    test_editing(model, x)
    # test_optimize_inverse(model, x)


if __name__ == '__main__':
    main()



