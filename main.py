# Press the green button in the gutter to run the script.
from animal_recognition.utils.read_dataset import ImageDataset

if __name__ == '__main__':
    image_dataset = ImageDataset(dataset_path='animal_recognition/image_dataset/',
                                 image_dimensions={'width': 256, 'height': 256})
    image_dataset.read_images()