import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def tensor2map(var):
    # if len(var.shape) == 4:
    #     var = var[0]
    if torch.is_tensor(var) and torch.max(var) < 1:
        mask = np.argmax(var.data.cpu().numpy(), axis=0)
    elif torch.is_tensor(var):
        mask = var.data.cpu().long().numpy()
    else:
        if len(var.shape) == 3:
            var = var[0]
        mask = var
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


def tensor2sketch(var):
    im = var[0].cpu().detach().numpy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = (im * 255).astype(np.uint8)
    return Image.fromarray(im)


# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def tensor2img(var):
    if len(var.shape) == 4:
        var = var[0]
    if not torch.is_tensor(var):
        return Image.fromarray(var)
    # var = var.clamp_(min=-1, max=1)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2img_np(var):
    if len(var.shape) == 4:
        var = var[0]
    if not torch.is_tensor(var):
        return Image.fromarray(var)
    # var = var.clamp_(min=-1, max=1)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var


def tensor2grayimg(var):
    assert len(var.shape) == 2
    if torch.is_tensor(var):
        var = var.cpu().detach().numpy()
#     var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'), 'L')


def img2tensor(img):
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    tensor = loader(img)
    return tensor


def numpy2img(var):
    return Image.fromarray(var.astype('uint8'))


def write2video(results_dir, *video_list):
    cat_video = None

    for video in video_list:
        video_numpy = video[:, :3, :, :].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array = []
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i])

    out_name = results_dir + '.mp4'
    _, height, width, layers = cat_video.shape
    size = (width, height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:, :, ::-1])
    out.release()
