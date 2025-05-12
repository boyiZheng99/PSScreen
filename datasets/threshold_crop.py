import argparse
import os
import numpy as np

import cv2
import PIL.Image



def get_bound(image, threshold=20, mode_keep=False, filename=''):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = [0, 0, 0, 0]
    for cnt in contours:
        res = cv2.boundingRect(cnt)
        if res[2] > result[2] or res[3] > result[3]:
            result = res
    x, y, w, h = result

    if mode_keep:
        if np.abs(image.shape[0] / image.shape[1] - 1.0) < 0.1:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    return x, y, w, h


def crop(image_path,
         output_image=None, output_log_file=None, thresh=20, mode_keep=False,
         image_suffix='.jpg', color_mode=False):
    os.makedirs(output_image, exist_ok=True)

    lines = []
    for root, dirs, files in os.walk(image_path):
        for i, image_file in enumerate(files):
            image = cv2.imread(os.path.join(image_path, image_file))
            ori_h, ori_w = image.shape[0], image.shape[1]
            x, y, w, h = get_bound(image, thresh, mode_keep, image_file)

            h0 = y
            h1 = y + h
            w0 = x
            w1 = x + w

            line = '{} {} {} {} {} {} {}\n'.format(
                image_file.replace(image_suffix, ''), h0, ori_h - h1, w0, ori_w - w1, ori_h, ori_w)
            lines.append(line)
            print(line, end='')

            image = image[h0:h1, w0:w1]
            cv2.imwrite(os.path.join(output_image, image_file), image)

    with open(output_log_file, mode='w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../retinal_dataset/data',
                        help='path of input root')
    parser.add_argument('--output', type=str, default='../retinal_dataset/data',
                        help='path of output root')
    args = parser.parse_args()

    # train
    crop(image_path=os.path.join(args.input, 'train/DDR_trainset'),
         output_image=os.path.join(args.output, 'train/DDR_trainset'),
         output_log_file=os.path.join(args.output, 'train_crop.txt'),
         thresh=20)

    # val
    crop(image_path=os.path.join(args.input, 'valid/DDR_validset'),
         output_image=os.path.join(args.output, 'valid/DDR_validset'),
         output_log_file=os.path.join(args.output, 'val_crop.txt'),
         thresh=30, mode_keep=True)

    # test
    crop(image_path=os.path.join(args.input, 'test/DDR_testset'),
         output_image=os.path.join(args.output, 'test/DDR_testset'),
         output_log_file=os.path.join(args.output, 'test_crop.txt'),
         thresh=20)