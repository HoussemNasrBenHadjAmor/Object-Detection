import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import numpy as np
import os
import glob as glob
import yaml
from tqdm import tqdm
from lxml import etree, objectify
import argparse
from xml.etree import ElementTree as et
from datasets import (create_valid_dataset)
import random
import gc

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
    CLASSES = data_configs['CLASSES']
    CLASSES_TO_AUGMENT = data_configs['CLASSES_TO_AUGMENT']    
    CLASS_MAPPING = data_configs['CLASS_MAPPING']    
    WIDTH = data_configs['WIDTH']    
    HEIGHT = data_configs['HEIGHT']
    NUMBER_OF_AUGMETATION_PER_IMAGE = data_configs['NUMBER_OF_AUGMETATION_PER_IMAGE']
    SAVE_DIR_EXAMPLES_PATH = data_configs['SAVE_DIR_EXAMPLES_PATH']
    
    
    augmentation_pipeline = A.Compose([
    #A.HorizontalFlip(p=0.4),
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.4),
    A.HorizontalFlip(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
    A.RandomCrop(width=min(WIDTH, 640), height=min(HEIGHT, 640), p=0.6),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),  # This standardizes the color scale
    #A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.4),
    #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
    #A.CLAHE(clip_limit=2, p=0.2),  # Add CLAHE for better contrast control
    #A.GaussianBlur(p=0.2),
    #A.HueSaturationValue(p=0.2),
    #A.OneOf([
     #   A.RandomRain(p=0.8),
     #   A.RandomSnow(p=0.8),
     #   A.RandomFog(p=1),
     #   A.RandomSunFlare(p=0.1)
    #], p=0.8),  # Add weather augmentation with a 30% probability
    ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))


    def save_augmented_image_and_labels(image, bboxes, class_labels, original_image_path, original_label_path, output_image_dir, output_label_dir, counter, class_mapping):
        image_name = os.path.splitext(os.path.basename(original_image_path))[0]
        augmented_image_name = f"{image_name}_aug_{counter}.jpg"
        augmented_label_name = f"{image_name}_aug_{counter}.xml"
    
        augmented_image_path = os.path.join(output_image_dir, augmented_image_name)
        augmented_label_path = os.path.join(output_label_dir, augmented_label_name)

        # Convert the tensor image back to a NumPy array and scale pixel values if necessary
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()

        # Ensure the image is in the range [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
        cv2.imwrite(augmented_image_path, image)
    
        # Create the XML annotation
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder(''),
            E.filename(augmented_image_path),
            E.path(augmented_image_path),
            E.source(
                E.database("Unknown")
            ),
            E.size(
                E.width(image.shape[1]),
                E.height(image.shape[0]),
                E.depth(image.shape[2])
            ),
            E.segmented(0),
        )
    
        for bbox, class_label in zip(bboxes, class_labels):
            xmin, ymin, xmax, ymax = bbox
            # Check and correct bounding box coordinates
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if int(xmin) == int(xmax):
                xmax += 1
            if int(ymin) == int(ymax):
                ymax += 1              
                
            class_name = class_mapping[int(class_label)]
            obj = E.object(
                E.name(class_name),
                E.pose("Unspecified"),
                E.truncated(0),
                E.difficult(0),
                E.occluded(0),
                E.bndbox(
                    E.xmin(int(xmin)),
                    E.ymin(int(ymin)),
                    E.xmax(int(xmax)),
                    E.ymax(int(ymax))
                )
            )
            anno_tree.append(obj)

        etree.ElementTree(anno_tree).write(augmented_label_path, pretty_print=True)

    def process_augmentation(base_dir, output_dir, classes_to_augment, augmentation_pipeline, width, height, classes, class_mapping, number_of_augmetation):
        aug_examples = []
       
        for folder in ['train', 'valid', 'test']:
            x= 0
            y=0
            image_dir = os.path.join(base_dir, folder)
            label_dir = os.path.join(base_dir, folder)
            
            print(f'We are in the {folder} folder to proccess with images')

            dataset = create_valid_dataset(image_dir, label_dir, width, height, classes)

            output_image_dir = os.path.join(output_dir, folder)
            output_label_dir = os.path.join(output_dir, folder)

            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)
           
            last_images_indices = list(range(max(0 , len(dataset))))
            #last_images_indices = list(range(max(0, len(dataset) - 200), len(dataset)))


            for index in tqdm(last_images_indices , desc=f'Processing augmetation for {folder} images'):
                try:
                    # Access the dataset item using the index
                    image, target = dataset[index]  # Correctly access the dataset using the index
                    image_path = target['image_path']
                    label_path = target['annot_path']

                    # Ensure the image is a NumPy array
                    if not isinstance(image, np.ndarray):
                        if torch.is_tensor(image):
                            image = image.permute(1, 2, 0).cpu().numpy()
                        else:
                            raise TypeError("Image must be a NumPy array or a torch tensor.")

                    # Convert the image to uint8 if it’s not
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)

                    if any(cls in target['labels'].tolist() for cls in classes_to_augment):
                        for i in range(number_of_augmetation):  # Number of augmentations per image
                            try:
                                augmented = augmentation_pipeline(image=image, bboxes=target['boxes'].tolist(), labels=target['labels'].tolist())
                                augmented_image = augmented['image']
                                augmented_bboxes = augmented['bboxes']
                                augmented_class_labels = augmented['labels']

                                # Convert augmented image to NumPy array if it's a tensor
                                if isinstance(augmented_image, torch.Tensor):
                                    augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()

                                if len(augmented_bboxes) > 0 and len(augmented_class_labels) > 0:
                                    if len(aug_examples) < 100:
                                        aug_examples.append(augmented_image)
                                    save_augmented_image_and_labels(augmented_image, augmented_bboxes, augmented_class_labels, image_path, label_path, output_image_dir, output_label_dir, i, class_mapping)
                                else:
                                    x+=1
                            except Exception as e:
                                print(f'Error during augmentation: {e}')
                    else :
                        y+=1

                    # Free memory for the original image and target after each iteration
                    del image, target    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f'Error during augmentation: {e}')

            print(f'Number of ignored images due to no labels or bboxes : {x}')
            print(f'Number of images which are not in the classes_to_augment : {y}')
        return aug_examples

    def save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH):
        os.makedirs(SAVE_DIR_EXAMPLES_PATH, exist_ok=True)  
        for i, image in enumerate(random_examples):

            if torch.is_tensor(image):
                image = image.cpu().detach().numpy()
                # Transpose dimensions from (C, H, W) to (H, W, C)
                image = np.transpose(image, (1, 2, 0))
            
            # Convert values to [0, 255] for uint8 images
            image = (np.clip(image * 255, 0, 255)).astype(np.uint8)
            # Save the image into path
            cv2.imwrite(os.path.join(SAVE_DIR_EXAMPLES_PATH, f'random_example_{i}.jpg'), image)
    
    aug_examples = []

    aug_array = process_augmentation(BASE_DIR, OUTPUT_DIR, CLASSES_TO_AUGMENT, augmentation_pipeline, WIDTH, HEIGHT, CLASSES, CLASS_MAPPING, NUMBER_OF_AUGMETATION_PER_IMAGE)

    aug_examples.extend(aug_array)

    # Select 10 random examples
    random_examples = random.sample(aug_examples, 10 if len(aug_examples) > 10 else len(aug_examples))

    save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

if __name__ == '__main__':
    args = parse_opt()
    main(args)
