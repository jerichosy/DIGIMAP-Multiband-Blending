import imageio
import numpy as np

import blending

import matplotlib.pyplot as plt


def read_image(path) -> np.ndarray:
    data = imageio.imread(path)
    return np.array(data) / 255


def read_mask(path) -> np.ndarray:
    image = read_image(path)
    return image.mean(axis=2)


def write_image(image: np.ndarray, path):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    imageio.imwrite(path, image)


def compute_num_levels(image: np.ndarray):
    h, w, c = image.shape
    return int(np.floor(np.log2(min(h, w))))


if __name__ == "__main__":
    target = read_image("target.jpg")
    source = read_image("source.jpg")
    mask = read_mask("mask.jpg")

    blender = blending.MultiBandBlending(num_levels=compute_num_levels(target))
    composite_multiband = blender(target, source, mask)
    write_image(composite_multiband, "multiband.jpg")

    blender = blending.NaiveBlending()
    composite_naive = blender(target, source, mask)
    write_image(composite_naive, "naive.jpg")

    plt.figure(figsize=(11, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(target)
    plt.title("Target")
    plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(source)
    plt.title("Source")
    plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(composite_multiband)
    plt.title("Multiband Blending")
    plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(composite_naive)
    plt.title("Naive Blending")
    plt.axis("off")
    plt.show()
    
