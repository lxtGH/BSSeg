import os
import os.path as osp
import cv2
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    sub_file_names = os.listdir(args.file1)
    os.makedirs(args.save_file, exist_ok=True)
    for sub_file in sub_file_names:
        if sub_file == '.DS_Store':
            continue
        images = os.listdir(osp.join(args.file1, sub_file))
        for image in images:
            if image == '.DS_Store':
                continue
            save_image_name = osp.join(args.save_file, sub_file+'_'+image)
            img1 = cv2.imread(osp.join(args.file1, sub_file, image))
            img2 = cv2.imread(osp.join(args.file2, sub_file, image))
            img = cv2.hconcat((img1, img2))
            cv2.imwrite(save_image_name, img)
            print(f'{image} done')
        print(f'{sub_file} done')


if __name__ == '__main__':
    main()