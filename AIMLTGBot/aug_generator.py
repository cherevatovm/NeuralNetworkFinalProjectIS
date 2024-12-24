import os
import random
import imageio
import imgaug.augmenters as iaa
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = imageio.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def save_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename, img in images:
        imageio.imwrite(os.path.join(folder, filename), img)

def fill_transparency_with_color(img, color):
    if img.shape[-1] == 4:
        alpha_channel = img[..., 3]
        img = img[..., :3]
        color_background = np.ones_like(img, dtype=np.uint8) * np.array(color, dtype=np.uint8)
        img = np.where(alpha_channel[..., None] == 0, color_background, img)
    return img

def augment_images(images, num_augmented=4, fill_color=(213, 181, 156)):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    rotation = iaa.Sequential([
        sometimes(iaa.Affine(
            rotate=(-40, 40),
            scale=(0.9, 1.0),
            mode='constant',
            cval=255
        )),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
    ], random_order=True)

    other = iaa.Sequential([
        sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
        sometimes(iaa.LinearContrast(1.2)),
        sometimes(iaa.Multiply((0.75, 1.25))),
    ], random_order=True)
    
    augmented_images = []
    for filename, image in images:
        image = fill_transparency_with_color(image, fill_color)
        for i in range(num_augmented):
            transformed_image = rotation(image=image)
            transformed_image = np.where(transformed_image == 255, np.array(fill_color, dtype=np.uint8), transformed_image)
            augmented_image = other(image=transformed_image)
            new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
            augmented_images.append((new_filename, augmented_image))
    return augmented_images

input_folder = 'clear_imgs'
output_folder = 'aug_imgs'

images = load_images_from_folder(input_folder)

augmented_images = augment_images(images)

save_images(augmented_images, output_folder)
