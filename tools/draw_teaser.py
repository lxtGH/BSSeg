# if __name__ == '__main__':
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

img_path = '/Users/hhe/research/mm2021_cenet/teaser/test_teaser.jpg'

mask = Image.open(img_path)
mask = np.array(mask)
mask[mask > 127] = 255
mask[mask <= 127] = 0
mask[mask == 255] = 1
mask = mask.astype(np.float32)
laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
mask = torch.from_numpy(mask)
mask = mask.unsqueeze(0)
boundary_masks = F.conv2d(mask.unsqueeze(1), laplacian_kernel, padding=1)
boundary_masks = boundary_masks.clamp(min=0)
boundary_masks[boundary_masks > 0.1] = 1
boundary_masks[boundary_masks <= 0.1] = 0

boundary_masks = boundary_masks.squeeze().cpu().numpy().astype('uint8')
mask = mask.squeeze()
mask = mask.cpu().numpy().astype('uint8')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

boundary_valid = boundary_masks == 1
dilate = cv2.dilate(mask, kernel=kernel)
contraction_mask = dilate - mask
contraction_mask[boundary_valid] = 1
contraction_mask_rgb = np.zeros([*contraction_mask.shape, 3])
contraction_mask_rgb[contraction_mask == 1] = [0, 0, 255]

erode = cv2.erode(mask, kernel=kernel)
expansion_mask = mask - erode
expansion_mask[boundary_valid] = 1
expansion_mask_rgb = np.zeros([*expansion_mask.shape, 3])
expansion_mask_rgb[expansion_mask == 1] = [255, 0, 0]

boundary_masks[boundary_masks == 1] = 255

contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundary = np.zeros_like(mask)
boundary = cv2.drawContours(boundary, contour, -1, 1, 2)
boundary = boundary.astype(np.float)
boundary[boundary == 1] = 255
cv2.imwrite('/Users/hhe/research/mm2021_cenet/teaser/contraction.jpg', contraction_mask_rgb)
cv2.imwrite('/Users/hhe/research/mm2021_cenet/teaser/expansion.jpg', expansion_mask_rgb)
cv2.imwrite('/Users/hhe/research/mm2021_cenet/teaser/boundary2.jpg', boundary_masks)
cv2.imwrite('/Users/hhe/research/mm2021_cenet/teaser/boundary.jpg', boundary)
