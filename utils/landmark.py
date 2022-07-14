import torch


def get_landmark_bbox(lm, scale=1):
    l_eye_id = [36, 42]
    r_eye_id = [42, 48]
    nose_id = [27, 36]
    mouth_id = [48, 68]
    p = 8
    bbox = []
    for _i, box_id in enumerate([mouth_id, l_eye_id, r_eye_id, nose_id]):
        box_lm = lm[:, box_id[0]:box_id[1]]
        ly, ry = torch.min(box_lm[:, :, 0], dim=1)[0], torch.max(box_lm[:, :, 0], dim=1)[0]
        lx, rx = torch.min(box_lm[:, :, 1], dim=1)[0], torch.max(box_lm[:, :, 1], dim=1)[0]  # shape: [b]
        lx, rx, ly, ry = (lx * scale).long(), (rx * scale).long(), (ly * scale).long(), (ry * scale).long()
        if _i == 1 or _i == 2:
            p = 15
        lx, rx, ly, ry = lx - p, rx + p, ly - p, ry + p
        lx, rx, ly, ry = lx.unsqueeze(1), rx.unsqueeze(1), ly.unsqueeze(1), ry.unsqueeze(1)
        bbox.append(torch.cat([ly, lx, ry, rx], dim=1))
    return bbox
