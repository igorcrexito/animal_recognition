from PIL import Image
import numpy as np
import random
import cv2


class AugmentData:

    @classmethod
    def flip_image(cls, image: Image, pts_list: list) -> (Image, list):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        for index, point in enumerate(pts_list):
            pts_list[index][0] = image.size[0]-point[0]
            pts_list[index][1] = point[1]

        return image, pts_list

    @classmethod
    def translate_image(cls, image: Image, pts_list: list) -> (Image, list):
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

        for index, point in enumerate(pts_list):
            pts_list[index][0] = point[0]+tx
            pts_list[index][1] = point[1]+ty

        return image, pts_list

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
