import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import shutil
import argparse
import yaml
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.ndimage import gaussian_filter



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
    WIDTH = data_configs['WIDTH']    
    HEIGHT = data_configs['HEIGHT']
    


    augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
    A.RandomCrop(width=min(WIDTH, 640), height=min(HEIGHT, 640), p=0.6),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
    #A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),  # This standardizes the color scale
    ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

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
        for folder in ['tarin', 'valid', 'test']:
            image_dir = os.path.join(base_dir, folder, 'images')
            label_dir = os.path.join(base_dir, folder, 'labels')

            output_image_dir = os.path.join(output_dir, folder, 'images')
            output_label_dir = os.path.join(output_dir, folder, 'labels')

            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            for label_file in tqdm(os.listdir(label_dir), desc=f'Processing augmetation for {folder} images'):
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
                            #aug_examples.append(augmented_image)
                            save_augmented_image_and_labels(augmented_image, augmented_bboxes, augmented_class_labels, image_path, label_path, output_image_dir, output_label_dir, i, class_names)

        return aug_examples
    
    def generate_haze (base_dir, num_images=800) :
        # Link to download the model checkpoint : https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true 

        # Initialize the model
        model_ckp  = '/teamspace/studios/this_studio/depth_anything_v2_vitl.pth' 

        model_depth = DepthAnythingV2(**model_configs[encoder])
        model_depth.load_state_dict(torch.load(model_ckp, map_location='cpu'))
        model_depth = model_depth.to(DEVICE).eval()

        np.random.seed(10)
        eps=0.15

        mean_r, std_r =  0.8, 0.005 ; r = np.random.normal(mean_r, std_r, 1)
        mean_g, std_g =  0.8, 0.005 ; g = np.random.normal(mean_g, std_g, 1)
        mean_b, std_b =  0.8, 0.005 ; b = np.random.normal(mean_b, std_b, 1) 
        mean = np.random.normal(3.25, 0.5, 1)

        atmospheric_light = np.array([r[0], g[0], b[0]])  # Assume white atmospheric light

        for folder in ['train', 'valid', 'test']:
            images_dir = os.path.join(base_dir, folder)
            # Get a list of all images in the directory
            image_paths = glob.glob(os.path.join(f'{images_dir}/images', '*.jpg'))

            # Randomly select 800 images (or fewer if the folder contains less than 800 images)
            if len(image_paths) > num_images:
                selected_image_paths = random.sample(image_paths, num_images)
            else:
                selected_image_paths = image_paths

            for image_path in tqdm(selected_image_paths, desc=f'Applying haze for {folder}'):
                rgb_img = cv2.imread(image_path)
                rgb_image_normalized = rgb_img.astype(np.float32) / 255.0
                depth_img = model_depth.infer_image(rgb_img) # HxW raw depth map in numpy
    
                depth_img_smoothed = gaussian_filter(depth_img, sigma=1.5)
        
                # Clip the values to a maximum of 255
                depth_img_clipped = np.clip(depth_img_smoothed, 0, 255).astype(np.uint8)

                inverted_depth_img = (1 - depth_img_clipped.astype(np.float32) / 250.0) + eps
        
                beta = np.random.normal(mean[0], 0.05, 1)
                beta[beta<0] = 0.01
                transmission_map = np.exp(-beta[0] * inverted_depth_img)
                # Compute the hazy image I(x) using the formula
                hazed_image = rgb_image_normalized * transmission_map[:, :, np.newaxis] + atmospheric_light * (1 - transmission_map[:, :, np.newaxis])
                hazed_image = (hazed_image * 255).astype(np.uint8)

                # Save the hazed image under the same path but append '_hazed'
                folder_path, file_name = os.path.split(image_path)
                file_name_without_ext, ext = os.path.splitext(file_name)
                hazed_image_path = os.path.join(folder_path, f"{file_name_without_ext}_hazed{ext}")
            
                cv2.imwrite(hazed_image_path, hazed_image)

                # Handle the associated .txt file
                txt_file_path = os.path.join(images_dir, 'labels', f"{file_name_without_ext}.txt")
                if os.path.exists(txt_file_path):
                    hazed_txt_path = os.path.join(images_dir, 'labels', f"{file_name_without_ext}_hazed.txt")
                    with open(txt_file_path, 'r') as txt_file:
                        txt_content = txt_file.read()
                    # Save the .txt file with '_hazed' appended
                    with open(hazed_txt_path, 'w') as hazed_txt_file:
                        hazed_txt_file.write(txt_content)
    
    aug_examples = []

    generate_haze(BASE_DIR)

    aug_array = process_augmentation(BASE_DIR, OUTPUT_DIR, augmentation_pipeline, CLASSES_TO_AUGMENT, NUMBER_OF_AUGMETATION_PER_IMAGE, CLASS_NAMES)

    #aug_examples.extend(aug_array)

    # Select 10 random examples
    #random_examples = random.sample(aug_examples, 10 if len(aug_examples) > 10 else len(aug_examples))

    #save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

    #print('Generation augmetation has completed successfully')


if __name__ == '__main__':
    args = parse_opt()
    main(args)