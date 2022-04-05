import argparse
import os
import os.path as osp
import imageio
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--img_height', type=int, default=360)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--fps', default=10, type=int)

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    images = os.listdir(args.image_path)
    images = sorted(images)
    gif_images = []
    for image in images:
        if image == '.DS_Store':
            continue
        img = imageio.imread(osp.join(args.image_path, image))
        img = cv2.resize(img, (args.img_width, args.img_height))
        gif_images.append(img)
    imageio.mimsave(args.save_path, gif_images, fps=args.fps)


if __name__ == '__main__':
    main()