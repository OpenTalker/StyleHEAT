import os
import torch


def get_delta(pca, latent, idx, strength):
    w_centered = latent - pca['mean'].to('cuda')
    lat_comp = pca['comp'].to('cuda')
    lat_std = pca['std'].to('cuda')
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta


def ganspace_edit(latents, pca, edit_directions):
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strength in edit_directions:
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = torch.zeros(latent.shape).to('cuda')
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta_padded)
    return torch.stack(edit_latents)


class LatentEditor(object):
    # Edit style code only without generating images
    def __init__(self):
        self.age_direction = None
        self.pose_direction = None
        self.ganspace_pca = None
        self.ganspace_directions = {
            # 'eyes': (54, 7, 8, 20),
            'beard': (58, 7, 9, -20),
            'lip': (34, 10, 11, 20)
        }

    def load(self, path_dic):
        interfacegan_root = path_dic['interfacegan']
        self.age_direction = torch.load(os.path.join(interfacegan_root, 'age.pt')).cuda()
        self.pose_direction = torch.load(os.path.join(interfacegan_root, 'pose.pt')).cuda()
        ganspace_path = path_dic['ganspace']
        self.ganspace_pca = torch.load(ganspace_path)

    # def apply_ganspace(self, latent, ganspace_pca, edit_directions):
    #     edit_latents = ganspace_edit(latent, ganspace_pca, edit_directions)
    #     return self._latents_to_image(edit_latents), edit_latents
    # 
    # def apply_interfacegan(self, latent, direction, factor=None):
    #     edit_latents = latent + factor * direction
    #     return self._latents_to_image(edit_latents), edit_latents

    def edit_style_code(self, wx, factor, choice):
        assert choice in ['young', 'old', 'beard', 'lip', 'pose']
        if choice in ['young', 'old']:
            # recommend factor is 5/-5
            edit_latents = wx + factor * self.age_direction
        elif choice == 'pose':
            edit_latents = wx + factor * self.pose_direction
        else:
            # recommend factor is 20/-20
            direction = self.ganspace_directions[choice]
            edit_direction = (direction[0], direction[1], direction[2], factor)
            edit_latents = ganspace_edit(wx, self.ganspace_pca, [edit_direction])
        return edit_latents

