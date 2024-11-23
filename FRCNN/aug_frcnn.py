import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
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
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from depth_anything_v2.dpt import DepthAnythingV2


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
    A.HorizontalFlip(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
    A.RandomCrop(width=min(WIDTH, 640), height=min(HEIGHT, 640), p=0.6),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),  # This standardizes the color scale
    ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    
    

    # take the colored image and its depth image as inputs and will generate a new image with an atmospheric fog effect
    def overlay_transparent_layer(rgb_image, grayscale_image):
        # Create a white layer with the same size as the input images
        white_layer = Image.new('RGBA', rgb_image.size, (216,216,216,0))
    
        # Convert images to numpy arrays for easier manipulation
        rgb_array = np.array(rgb_image)
        grayscale_array = np.array(grayscale_image)
        white_array = np.array(white_layer)
    
        # Calculate alpha values (invert grayscale values)
        alpha = 255 - grayscale_array
    
        # Set the alpha channel of the white layer
        white_array[:, :, 3] = alpha
    
        # Convert back to PIL Image
        white_layer_transparent = Image.fromarray(white_array, 'RGBA')
    
        # Composite the images
        result = Image.alpha_composite(rgb_image.convert('RGBA'), white_layer_transparent)
    
        return result                

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
        for folder in ['train', 'valid', 'test']:
            x= 0
            y=0
            image_dir = os.path.join(base_dir, folder)
            label_dir = os.path.join(base_dir, folder)

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
                                    #if len(aug_examples) < 100:
                                     #   aug_examples.append(augmented_image)
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
    
    def apply_haze_and_save_after_applying_and_saving_augmentation(base_dir):
        # depth estimation using a pre-trained model
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

        for folder in ['train', 'valid', 'test']:
            images_dir = os.path.join(base_dir, folder)
            #print(f'images_dir : {images_dir}')
            # Get a list of all images in the directory
            image_paths = glob.glob(os.path.join(images_dir, '*.jpg')) 
            #print(f'image_paths : {image_paths}')
            for image_path in tqdm(image_paths, desc=f'Applying haze for {folder}'):
                try:
                    # Get the original_image
                    original_image = Image.open(image_path).convert("RGBA")
                    # perform depth estimation
                    depth = pipe(original_image)["depth"]

                    # reduce saturation
                    enhancer = ImageEnhance.Color(original_image)
                    img_2 = enhancer.enhance(0.5)
                    
                    # reduce brightness
                    enhancer2 = ImageEnhance.Brightness(img_2)
                    img_2 = enhancer2.enhance(0.7)
                    
                    # increase contrast
                    enhancer3 = ImageEnhance.Contrast(img_2)
                    img_2 = enhancer3.enhance(2.2)

                    #pass the input image (after color adjustments) and the depth image to the function overlay_transparent_layer to add the haze.
                    result_img = overlay_transparent_layer(img_2, depth)

                    # Convert result image to RGB before saving as JPEG
                    result_img = result_img.convert("RGB")

                    # remove the old img 
                    os.remove(image_path)

                    # save the new img 
                    result_img.save(image_path)


                except Exception as e:
                    print(f'Error during haze application for {image_path}: {e}')
                    
    def generate_haze (base_dir, num_images=400) :
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
            image_paths = glob.glob(os.path.join(images_dir, '*.jpg')) 

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

                # Handle the associated XML file
                xml_file_path = os.path.join(folder_path, f"{file_name_without_ext}.xml")
                if os.path.exists(xml_file_path):
                    hazed_xml_path = os.path.join(folder_path, f"{file_name_without_ext}_hazed.xml")
                    with open(xml_file_path, 'r') as xml_file:
                        xml_content = xml_file.read()
                    # Save the XML file with '_hazed' appended
                    with open(hazed_xml_path, 'w') as hazed_xml_file:
                        hazed_xml_file.write(xml_content)

    
    generate_haze(BASE_DIR)

    process_augmentation(BASE_DIR, OUTPUT_DIR, CLASSES_TO_AUGMENT, augmentation_pipeline, WIDTH, HEIGHT, CLASSES, CLASS_MAPPING, NUMBER_OF_AUGMETATION_PER_IMAGE)

    #aug_examples.extend(aug_array)

    # Select 10 random examples
    #random_examples = random.sample(aug_examples, 10 if len(aug_examples) > 10 else len(aug_examples))

    #save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

    #apply_haze_and_save_after_applying_and_saving_augmentation(BASE_DIR)
    print('Generation augmetation has completed successfully')

    

if __name__ == '__main__':
    args = parse_opt()
    main(args)
