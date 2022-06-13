import os
import re
from itertools import product
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt

image_ext = re.compile('.*\.(jpg|jpeg|png)$')


def split(image: np.ndarray) -> tuple:
    input_height, input_width, __ = image.shape
    # magic numbers
    rs = [500, input_height // 2]
    cs = [500, input_width // 3 + 250, 2 * input_width // 3]
    output_height = output_width = 1300

    return tuple(image[r:r + output_height, c:c + output_width] for r, c in product(rs, cs))


def split_all(load_path: str) -> Generator:
    for filename in os.listdir(load_path):
        if not image_ext.search(filename):
            continue
        image_path = os.path.join(load_path, filename)
        input_image = plt.imread(image_path)
        for output_image in split(input_image):
            yield output_image


def test_split():
    image_path = os.path.join('.', 'data', 'beet_armyworm_raw', '0522', 'P1030448.jpg')
    image = plt.imread(image_path)
    plt.imshow(image)
    fig, axs = plt.subplots(2, 3)
    for ax, img in zip(axs.ravel(), split(image)):
        ax.axis('off')
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def test_split_all():
    images_path = os.path.join('.', 'data', 'beet_armyworm_raw')
    save_path = os.path.join(images_path, 'beet_armyworm')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for path, *__ in os.walk(images_path):
        if os.path.realpath(path) == os.path.realpath(save_path):
            continue
        dirname = os.path.split(path)[-1]
        for i, output_image in enumerate(split_all(path), 1):
            prefix, suffix = divmod(i, 6)
            output_filename = f"{dirname}_{prefix:04d}{suffix}.jpg"
            plt.imsave(os.path.join(save_path, output_filename), output_image)


if __name__ == '__main__':
    test_split()
    test_split_all()
