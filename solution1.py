from typing import List
from cv2 import cv2
import argparse
import os
import numpy as np

def read_images(path: str) -> List[np.ndarray]:
    images = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        image = cv2.imread(file_path)
        images.append(image)
    return images

def hist_equalize(image: np.ndarray) -> np.ndarray:
    image_copy = np.copy(image)
    height, width = image_copy.shape[:2]
    image_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    [h, s, v] = cv2.split(image_hsv)
    n_k = [0] * 256
    for pixel in v.flatten():
        n_k[pixel] += 1
    p_r = [0.] * 256
    for i, n in enumerate(n_k):
        p_r[i] = n / (height * width)
    s_k = [0.] * 256
    s_k[0] = 255 * p_r[0]
    for i, p in enumerate(p_r[1:]):
        s_k[i + 1] = 255 * p + s_k[i]
    s_k = list(map(lambda x: round(x), s_k))
    v_new = np.copy(v)
    for i in range(v_new.shape[0]):
        for j in range(v_new.shape[1]):
            v_new[i, j] = s_k[v_new[i, j]]
    eq_image = np.dstack((h, s, v_new))
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_HSV2BGR)
    return eq_image

def adap_hist_equalize(image: np.ndarray) -> np.ndarray:
    image_copy = np.copy(image)
    ht, wd = image_copy.shape[:2]
    h_step, w_step = (64, 64)
    for h in range(0, ht, h_step):
        for w in range(0, wd, w_step):
            image_copy[h:h+h_step, w:w+w_step, :] = hist_equalize(image_copy[h:h+h_step, w:w+w_step, :])
    image_copy = cv2.GaussianBlur(image_copy, (5, 5), cv2.BORDER_DEFAULT)
    return image_copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--images_path', type=str,
                        default='./data/adaptive_hist_data',
                        help='The path where photos are stored.')
    parser.add_argument('-op', '--output_path', type=str,
                        default='./data/outputs/',
                        help='The path where outputs are stored.')
    args = parser.parse_args()
    images_path = args.images_path
    output_path = args.output_path

    if not os.path.exists(images_path):
        raise ValueError(f'The path {images_path} does not exist!')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    images = read_images(images_path)

    hist_eq_images_path = os.path.join(output_path, 'hist_eq_images')
    if not os.path.exists(hist_eq_images_path):
        os.makedirs(hist_eq_images_path)
    for i, image in enumerate(images):
        eq_image = hist_equalize(image)
        eq_image_path = os.path.join(hist_eq_images_path, f'image_{i}.png')
        cv2.imwrite(eq_image_path, eq_image)

    adap_hist_eq_images_path = os.path.join(output_path, 'adap_hist_eq_images')
    if not os.path.exists(adap_hist_eq_images_path):
        os.makedirs(adap_hist_eq_images_path)
    for i, image in enumerate(images):
        eq_image = adap_hist_equalize(image)
        eq_image_path = os.path.join(adap_hist_eq_images_path, f'image_{i}.png')
        cv2.imwrite(eq_image_path, eq_image)
