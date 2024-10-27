import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import shutil
import argparse
import yaml



def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    args = vars(parser.parse_args())
    return args


def main (args):
    # Initialize W&B with project name.
    # wandb_init(name=args['project_name'])
    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Settings/parameters/constants.
    BASE_DIR = data_configs['BASE_DIR']
    OUTPUT_DIR = data_configs['OUTPUT_DIR']
    CLASSES_TO_AUGMENT = data_configs['CLASSES_TO_AUGMENT']    
    NUMBER_OF_AUGMETATION_PER_IMAGE = data_configs['NUMBER_OF_AUGMETATION_PER_IMAGE']
    SAVE_DIR_EXAMPLES_PATH = data_configs['SAVE_DIR_EXAMPLES_PATH']
    CLASS_NAMES = data_configs['CLASS_NAMES']
    

    # Define augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomCrop(width=450, height=450, p=0.4),
        A.GaussianBlur(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
        A.OneOf([
            A.RandomRain(p=0.2),
            A.RandomSnow(p=0.2),
            A.RandomFog(p=0.2),
            A.RandomSunFlare(p=0.2)
        ], p=0.5),  # Add weather augmentation with a 30% probability
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Get the image, bboxes and class label for each image in the dataset
    def load_image_and_labels(image_path, label_path, class_names):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        with open(label_path, 'r') as file:
            labels = file.readlines()
        bboxes = []
        class_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.split())
            class_labels.append(class_names[int(class_id)])
            bboxes.append([x_center, y_center, width, height])

        return image, bboxes, class_labels, h, w
    
    # Save the augmented label and image
    def save_augmented_image_and_labels(image, bboxes, class_labels, original_image_path, original_label_path, output_image_dir, output_label_dir, counter, class_names):
        image_name = os.path.splitext(os.path.basename(original_image_path))[0]
        label_name = os.path.splitext(os.path.basename(original_label_path))[0]

        augmented_image_path = os.path.join(output_image_dir, f"{image_name}_aug_{counter}.jpg")
        augmented_label_path = os.path.join(output_label_dir, f"{label_name}_aug_{counter}.txt")

        # Convert the tensor image back to a NumPy array
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()


        cv2.imwrite(augmented_image_path, image)
    
        with open(augmented_label_path, 'w') as file:
            for bbox, class_label in zip(bboxes, class_labels):
                class_id = class_names.index(class_label)
                file.write(f"{class_id} " + " ".join(map(str, bbox)) + "\n")

    def save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH):
        os.makedirs(SAVE_DIR_EXAMPLES_PATH, exist_ok=True)  
        for i, image in enumerate(random_examples):
            tensor = image.cpu().detach()
             # Remove the batch dimension and convert to NumPy array
            image = tensor.squeeze().numpy()
            # Transpose dimensions from (C, H, W) to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            # Convert values to [0, 255] for uint8 images
            image = (np.clip(image * 255, 0, 255)).astype(np.uint8)
            # Save the image into path
            cv2.imwrite(os.path.join(SAVE_DIR_EXAMPLES_PATH, f'random_example_{i}.jpg'), image)

    def process_augmentation(base_dir, output_dir, augmentation_pipeline, classes_to_augment, NUMBER_OF_AUGMETATION_PER_IMAGE, class_names):
        aug_examples = []
        # Process each directory
        for folder in ['train', 'valid', 'test']:
            image_dir = os.path.join(base_dir, folder, 'images')
            label_dir = os.path.join(base_dir, folder, 'labels')

            output_image_dir = os.path.join(output_dir, folder, 'images')
            output_label_dir = os.path.join(output_dir, folder, 'labels')

            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            for label_file in tqdm(os.listdir(label_dir)):
                label_path = os.path.join(label_dir, label_file)
                image_path = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))

                image, bboxes, class_labels, h, w = load_image_and_labels(image_path, label_path, class_names)

                if any(cls in class_labels for cls in classes_to_augment):
                    for i in range(NUMBER_OF_AUGMETATION_PER_IMAGE):  # Number of augmentations per image
                        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                        augmented_image = augmented['image']
                        augmented_bboxes = augmented['bboxes']
                        augmented_class_labels = augmented['class_labels']

                        if augmented_bboxes and augmented_class_labels:
                            aug_examples.append(augmented_image)
                            save_augmented_image_and_labels(augmented_image, augmented_bboxes, augmented_class_labels, image_path, label_path, output_image_dir, output_label_dir, i, class_names)

        return aug_examples
    
    aug_examples = []

    aug_array = process_augmentation(BASE_DIR, OUTPUT_DIR, augmentation_pipeline, CLASSES_TO_AUGMENT, NUMBER_OF_AUGMETATION_PER_IMAGE, CLASS_NAMES)

    aug_examples.extend(aug_array)

    # Select 10 random examples
    random_examples = random.sample(aug_examples, 10 if len(aug_examples) > 10 else len(aug_examples))

    save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

    print('Generation augmetation has completed successfully')

if __name__ == '__main__':
    args = parse_opt()
    main(args)