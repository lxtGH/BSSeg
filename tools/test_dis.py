import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

def gauss_pro(v, sigma):
    part1 = 1 / (sigma * np.sqrt(2 * np.pi))
    part2 = np.exp(-1 * v ** 2 / (2 * sigma ** 2))
    ret = part1 * part2
    return np.around(ret / max(ret), 1)
    # return part1 * part2

if __name__ == '__main__':


    mask = Image.open('/Users/hhe/research/dataset/Trans10k/train/graymasks/6333_maskgray.png')
    mask = np.array(mask)
    mask = cv2.resize(mask, (512, 512))
    mask_255 = mask.copy()
    mask_255[mask_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/orimask.jpg', mask_255)
    mask_numpy = mask.copy()
    mask = torch.tensor(mask).float()

    contours, _ = cv2.findContours(mask_numpy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_numpy)
    boundary = cv2.drawContours(boundary, contours, -1, 1, 2)
    boundary_255 = boundary.copy()
    boundary_255[boundary_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/boundary.jpg', boundary_255)
    concat_contours = []
    for contour in contours:
        concat_contours.append(contour.squeeze(1))
    concat_contours = np.vstack(concat_contours)                # x, y

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=mask.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_targets = F.conv2d(mask.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    boundary_targets_255 = boundary_targets.clone().squeeze().numpy()
    boundary_points = np.nonzero(boundary_targets_255)
    boundary_points = np.hstack((boundary_points[1][:, None], boundary_points[0][:, None]))   # x, y
    boundary_targets_255[boundary_targets_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/boundary_la.jpg', boundary_targets_255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilate = cv2.dilate(mask_numpy, kernel=kernel)
    dilate = dilate - mask_numpy
    dilate_255 = dilate.copy()
    dilate_255[dilate_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/dilate.jpg', dilate_255)

    erode = cv2.erode(mask_numpy, kernel=kernel)
    erode = mask_numpy - erode
    erode_255 = erode.copy()
    erode_255[erode_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/erode.jpg', erode_255)

    open = cv2.morphologyEx(mask_numpy, cv2.MORPH_OPEN, kernel, iterations=1)
    open = mask_numpy - open
    open_255 = open.copy()
    open_255[open_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/open.jpg', open_255)

    close = cv2.morphologyEx(mask_numpy, cv2.MORPH_CLOSE, kernel, iterations=1)
    close = close - mask_numpy
    close_255 = close.copy()
    close_255[close_255 > 0] = 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/close.jpg', close_255)

    # for dilate
    dilate_valid_points = np.nonzero(dilate)               # y, x
    erode_valid_points = np.nonzero(erode)                 # y, x
    dilate_dis = []
    erode_dis = []
    for i in range(len(erode_valid_points[0])):
        # temp_dis = 10000
        # for contour in contours:
        #     temp_dis = min(abs(cv2.pointPolygonTest(contour, (erode_valid_points[1][i], erode_valid_points[0][i]), True)), temp_dis)
        # erode_dis.append(temp_dis)
        # erode_dis.append(
        #     abs(cv2.pointPolygonTest(boundary_points, (erode_valid_points[1][i], erode_valid_points[0][i]), True)))
        erode_dis.append(abs(cv2.pointPolygonTest(concat_contours, (erode_valid_points[1][i], erode_valid_points[0][i]), True)))
    erode_dis_max = max(erode_dis)
    erode_dis_min = min(erode_dis)
    gauss_erode = gauss_pro(np.array(erode_dis), sigma=(erode_dis_max - erode_dis_min) / 3)

    for i in range(len(dilate_valid_points[0])):
        # for contour in contours:
        #     temp_dis = min(abs(cv2.pointPolygonTest(contour, (dilate_valid_points[1][i], dilate_valid_points[0][i]), True)), temp_dis)
        # dilate_dis.append(temp_dis)
        dilate_dis.append(abs(cv2.pointPolygonTest(concat_contours, (dilate_valid_points[1][i], dilate_valid_points[0][i]), True)))
        # dilate_dis.append(abs(cv2.pointPolygonTest(boundary_points, (dilate_valid_points[1][i], dilate_valid_points[0][i]), True)))

    dilate_dis_max = max(dilate_dis)
    dilate_dis_min = min(dilate_dis)
    gauss_dilate = gauss_pro(np.array(dilate_dis), sigma=(dilate_dis_max - dilate_dis_min) / 3)

    erode_gray = erode.copy().astype(np.float)
    dilate_gray = dilate.copy().astype(np.float)
    erode_gray[erode_valid_points[0], erode_valid_points[1]] = gauss_erode
    dilate_gray[dilate_valid_points[0], dilate_valid_points[1]] = gauss_dilate
    erode_gray = erode_gray * 255
    dilate_gray = dilate_gray * 255
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/erode_gray.jpg', erode_gray)
    cv2.imwrite('/Users/hhe/research/dataset/Trans10k/train/graymasks/test/dilate_gray.jpg', dilate_gray)
