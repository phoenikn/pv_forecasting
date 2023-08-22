import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from einops import rearrange


from vit.vit import TimeVIT
from vit.vit_only_vis import VIT_VIS

np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_size = (258, 258)

transform = transforms.Compose([
                transforms.CenterCrop((1500, 1500)),
                transforms.Resize(tensor_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1119, 0.1167, 0.1461),
                                     (0.116, 0.1160, 0.1288))
            ])


def visualize_grid_attention_v2(img_path, attention_mask, ratio=1.0, cmap="jet"):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    w, h = img.size
    left = (w - 1500) / 2
    top = (h - 1500) / 2
    right = (w + 1500) / 2
    bottom = (h + 1500) / 2
    img = img.crop((left, top, right, bottom))
    img_h, img_w = img.size[0], img.size[1]
    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(0.04 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
    axis[0].imshow(img, alpha=1)
    plt.margins(0, 0)
    # plt.savefig("map_with_attn.eps", format="eps", dpi=10)


with torch.no_grad():
    model = TimeVIT()
    model.load_state_dict(torch.load("VIT.pth", map_location=device))
    model.to(device)
    model.eval()
    state = model.state_dict()
    state_keys = list(state.keys())

    model_vis = VIT_VIS()
    state_vis = model_vis.state_dict().copy()
    vis_keys = list(state_vis.keys())

    for i in range(len(state_vis)):
        state_vis[vis_keys[i]] = state[state_keys[i]]

    model_vis.load_state_dict(state_vis)


    def show_mask(name):
        img1 = Image.open(name)
        x = transform(img1)
        x.size()
        _, att = model_vis(x.unsqueeze(0))
        att = att[:, :]
        for layer in att:
            for head in layer:
                head = (head - head.min()) / (head.max() - head.min())

        att = torch.mean(att, 0)
        att = torch.mean(att, 0)
        # att = att[0, 1:]
        att = torch.mean(att, 0)[1:]
        att = rearrange(att, "(h w) -> h w", h=16)
        att = np.array(att)

        visualize_grid_attention_v2(name, att)
        plt.show()

    show_mask("test.png")