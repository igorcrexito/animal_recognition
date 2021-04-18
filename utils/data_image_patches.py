from PIL.Image import Image
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np


class ImagePatches:

    def __init__(self, input_image: Image, list_of_points: list):
        self.input_image = input_image
        self.list_of_points = list_of_points
        self.img_counter = 0

    def create_patch_list(self, patch_size: int):
        image_patches = []

        for point in self.list_of_points:
            image_patch = self.input_image.crop((point[0]-patch_size,
                                                 point[1]-patch_size,
                                                 point[0]+patch_size,
                                                 point[1]+patch_size))

            image_patches.append(image_patch)

        return image_patches
