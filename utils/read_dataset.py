from PIL import Image, ImageDraw
import glob
from pathlib import Path
from typing import Union
import numpy as np
import IPython

from utils.data_augmentation import AugmentData


class ImageDataset:

    def __init__(self, dataset_path: str, image_dimensions: dict):
        self.image_list = []
        self.dataset_path = dataset_path
        self.image_dimensions = image_dimensions

    def read_images(self):

        # assembling a list of images
        image_list = []
        fiducial_points = []
        image_class_list = []

        # we must resize the images and their corresponding coordinates
        image_size = (self.image_dimensions['width'], self.image_dimensions['height'])

        # iterating over images and keypoints
        for image_name in glob.glob(self.dataset_path + "/*.jpg"):
            try:
                pts_file = image_name[:-4] + '.pts'

                image = Image.open(image_name)

                # resizing images and adjusting fiducial coordinates
                dx_coefficient = float(self.image_dimensions['width']) / image.size[0]
                dy_coefficient = float(self.image_dimensions['height']) / image.size[1]
                image = image.resize(image_size)
                pts_vector = self.adjust_fiducial_points(pts_vector=self.read_pts(pts_file),
                                                         dx_coefficient=dx_coefficient,
                                                         dy_coefficient=dy_coefficient)

                image_list.append(image)
                fiducial_points.append(pts_vector)
                image_class_list.append(image_name[14:].split('_')[0])

                # insert augmented images
                flipped_tuple = AugmentData.flip_image(image=image, pts_list=pts_vector.copy())
                translated_tuple = AugmentData.translate_image(image=image, pts_list=pts_vector.copy())

                image_list.append(flipped_tuple[0])
                image_class_list.append(image_name[14:].split('_')[0])
                image_list.append(translated_tuple[0])
                image_class_list.append(image_name[14:].split('_')[0])
                fiducial_points.append(flipped_tuple[1])
                fiducial_points.append(translated_tuple[1])

            except:
                print('invalid image')

        return image_list, fiducial_points, image_class_list

    def adjust_fiducial_points(self, pts_vector, dx_coefficient, dy_coefficient):
        #d = ImageDraw.Draw(image, 'RGBA')
        for index, point in enumerate(pts_vector):
            pts_vector[index][0] = round(point[0] * dx_coefficient)
            pts_vector[index][1] = round(point[1] * dy_coefficient)

            #shape = [(pts_vector[index][0], pts_vector[index][1]),
            #        (pts_vector[index][0] + 10, pts_vector[index][1] + 10)]

            #d.ellipse(shape, fill="yellow", outline="red")

        #image.show()
        return pts_vector

    def read_pts(self, filename: Union[str, bytes, Path]) -> np.ndarray:
        """Read a .PTS landmarks file into a numpy array"""
        with open(filename, 'rb') as f:
            # process the PTS header for n_rows and version information
            rows = version = None
            for line in f:
                if line.startswith(b"//"):  # comment line, skip
                    continue
                header, _, value = line.strip().partition(b':')
                if not value:
                    if header != b'{':
                        raise ValueError("Not a valid pts file")
                    if version != 1:
                        raise ValueError(f"Not a supported PTS version: {version}")
                    break
                try:
                    if header == b"n_points":
                        rows = int(value)
                    elif header == b"version":
                        version = float(value)  # version: 1 or version: 1.0
                    elif not header.startswith(b"image_size_"):
                        # returning the image_size_* data is left as an excercise
                        # for the reader.
                        raise ValueError
                except ValueError:
                    raise ValueError("Not a valid pts file")

            # if there was no n_points line, make sure the closing } line
            # is not going to trip up the numpy reader by marking it as a comment
            points = np.loadtxt(f, max_rows=rows, comments="}")

        if rows is not None and len(points) < rows:
            raise ValueError(f"Failed to load all {rows} points")
        return points
