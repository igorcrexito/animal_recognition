# Press the green button in the gutter to run the script.
from utils.read_dataset import ImageDataset
from sklearn import preprocessing


def preparing_dataset_classes(dataset_classes_list: list) -> list:
    lb = preprocessing.LabelBinarizer()
    lb.fit(dataset_classes_list)

    return lb.transform(image_class_list)


if __name__ == '__main__':
    image_dataset = ImageDataset(dataset_path='image_dataset/',
                                 image_dimensions={'width': 256, 'height': 256})
    _, _, image_class_list = image_dataset.read_images()
    image_class_list = preparing_dataset_classes(image_class_list)
