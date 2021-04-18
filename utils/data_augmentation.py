from PIL import Image
import numpy as np
import random
import cv2


class AugmentData:

    # TODO adjustment of coordinates of fiduciary points

    @classmethod
    def flip_image(cls, image: Image) -> Image:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    @classmethod
    def translate_image(cls, image: Image) -> Image:
        image = np.asarray(image)

        img_width = np.shape(image)[1]
        img_height = np.shape(image)[0]

        t1 = int(img_width * 0.1)
        t2 = int(img_height * 0.1)

        tx = random.randint(-t1, t1)
        ty = random.randint(-t2, t2)

        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, translation_matrix, (img_width, img_height))

        image = Image.fromarray(image)

        return image

    @classmethod
    def zooming(cls, image: Image) -> Image:
        img = np.asarray(image)

        img_width = np.shape(img)[1]
        img_height = np.shape(img)[0]

        t1 = int(img_width * 0.1)
        t2 = int(img_height * 0.1)

        tx = random.randint(0, t1)
        ty = random.randint(0, t2)

        img = img[ty:img_height - ty, tx:img_width - tx, :]

        img = Image.fromarray(img)
        img = img.resize((img_width, img_height))

        return img
