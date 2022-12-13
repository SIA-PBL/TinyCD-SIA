import os
from os.path import join, basename, normpath
from split_image import split_image


def crop_fbf(data_path: str, mode: str):
    dir_path_image = join(data_path, mode, "images")
    dir_path_label = join(data_path, mode, "labels")
    dir_output_image = join(data_path, mode, "images_256")
    dir_output_label = join(data_path, mode, "labels_256")
    for (root, directories, files) in os.walk(dir_path_image):
        image_dir_name = os.path.basename(os.path.normpath(root))
        os.system(f'split-image {root} 4 4 --output-dir ./SPN7/{mode}/images_256/{image_dir_name}')
    for (root, directories, files) in os.walk(dir_path_label):
        image_dir_name = os.path.basename(os.path.normpath(root))
        os.system(f'split-image {root} 4 4 --output-dir ./SPN7/{mode}/labels_256/{image_dir_name}')


if __name__ == '__main__':
    crop_fbf('/home/jovyan/TinyCD-SIA/dataset/SPN7', 'train')
    crop_fbf('/home/jovyan/TinyCD-SIA/dataset/SPN7', 'val')
