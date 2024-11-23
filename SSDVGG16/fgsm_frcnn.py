import torch
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from models.create_fasterrcnn_model import create_model
#from torchvision.transforms import functional as F
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree, objectify
import yaml
import argparse
import os
from tqdm import tqdm
import cv2
import uuid
import random
from datasets import (create_valid_dataset)


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='path to the model'
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
    MODEL_PATH = data_configs['MODEL_PATH']
    BASE_DIR = data_configs['BASE_DIR']
    OUTPUT_DIR = data_configs['OUTPUT_DIR']
    EPSILONS = data_configs['EPSILONS']
    CLASSES = data_configs['CLASSES']   
    CLASS_MAPPING = data_configs['CLASS_MAPPING']    
    WIDTH = data_configs['WIDTH']    
    HEIGHT = data_configs['HEIGHT']
    SAVE_DIR_EXAMPLES_PATH = data_configs['SAVE_DIR_EXAMPLES_PATH']
    MODEL_NAME = data_configs['MODEL_NAME']
    SIZE = data_configs['SIZE']
    
    use_cuda=True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    print('Loading pretrained weights...')
    # Load the pretrained checkpoint.
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    #print(f'checkpoint : {checkpoint}')
    ckpt_state_dict = checkpoint['model_state_dict']
    #print(f'ckpt_state_dict : {ckpt_state_dict}')
    # Get the classes and classes number from the checkpoint
    NUM_CLASSES = checkpoint['config']['NC']

    # Build the new model with number of classes same as checkpoint.
    build_model = create_model[MODEL_NAME]
    model = build_model(num_classes=NUM_CLASSES, size=SIZE)
    # Load weights.
    model.load_state_dict(ckpt_state_dict)

    # Set the model to device and evaluation mode
    model.to(device).eval()

    # Define the loss function
    def loss_fn(outputs, target, device):
        #scores = outputs[0]['scores']
        predicted_boxes = outputs[0]['boxes']  # Predicted bounding boxes
        predicted_labels = outputs[0]['labels'].float()  # Predicted class labels
        predicted_scores = outputs[0]['scores']  # Confidence scores

        ground_truth_boxes = target['boxes'].to(device)
        ground_truth_labels = target['labels'].float().to(device)
        ground_truth_scores = torch.ones(target['labels'].size(0), device=device)

        bbox_loss = F.smooth_l1_loss(predicted_boxes, ground_truth_boxes)
        obj_loss = F.binary_cross_entropy_with_logits(predicted_scores, ground_truth_scores)
        class_loss = F.binary_cross_entropy_with_logits(predicted_labels, ground_truth_labels)

        # Total loss
        total_loss = bbox_loss + obj_loss + class_loss
        return total_loss    
    
    # FGSM function to generate adversarial examples
    def fgsm_attack(image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def check_pred_target(output , target ,skip):
        output_shape = output[0]['labels'].shape[0]
        target_shape = target['labels'].shape[0]
        if output_shape != target_shape:
            if skip:
                return None
            else :
                return ''   
        else:
            if skip:
                return ''
            else:
                return None
            
    def save_fgsm_image_label(output_dir, image, target, folder, class_mapping):
        tensor = image.cpu().detach()
        # Remove the batch dimension and convert to NumPy array
        image = tensor.squeeze().numpy()
        # Transpose dimensions from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        # Convert values to [0, 255] for uint8 images
        image = (np.clip(image * 255, 0, 255)).astype(np.uint8)

        image_dir = os.path.join(output_dir, folder)

        os.makedirs(image_dir, exist_ok=True)  
    
        # Generate a random image_name using UUID    
        random_filename = str(uuid.uuid4())
        image_name = random_filename + '_fgsm.jpg'
        label_name = random_filename + '_fgsm.xml'
        fsgm_image_path = os.path.join(image_dir, image_name)    
        fsgm_label_path = os.path.join(image_dir, label_name)    

        cv2.imwrite(fsgm_image_path, image)

        bboxes = target['boxes']
        class_labels = target['labels']

        # Create the XML annotation
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder(''),
            E.filename(fsgm_image_path),
            E.path(fsgm_image_path),
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

        etree.ElementTree(anno_tree).write(fsgm_label_path, pretty_print=True)        
    
    def create_adversarial_attack(epsilon, base_dir, output_dir, width, height, classes, class_mapping, device, number_max_fgsm=5000):
        """
        Generates a fixed number of adversarial attacks from random images in the dataset.

        Args:
            epsilon (float): Perturbation magnitude.
            base_dir (str): Base directory of the dataset.
            output_dir (str): Directory to save adversarial images.
            width (int): Image width.
            height (int): Image height.
            classes (list): List of class names.
            class_mapping (dict): Mapping of class indices to names.
            device (torch.device): PyTorch device.
            number_max_fgsm (int): Maximum number of adversarial attacks to generate.
        """
        

        for folder in ['train', 'valid', 'test']:
            i = 0  # Counter for generated adversarial examples
            correct = 0
            image_dir = os.path.join(base_dir, folder)
            label_dir = os.path.join(base_dir, folder)
            dataset = create_valid_dataset(image_dir, label_dir, width, height, classes)

            # Randomly select a subset of the dataset
            dataset_indices = list(range(len(dataset)))
            random_indices = random.sample(dataset_indices, min(len(dataset), len(dataset)))

            # Process only the randomly selected indices
            for idx in tqdm(random_indices, desc=f'Processing adversarial attack in {folder} images'):
                # Check if we have reached the max adversarial examples
                if i >= number_max_fgsm:
                    print(f"Reached the limit of {number_max_fgsm} adversarial examples for {folder}.")
                    break

                image, target = dataset[idx]
                image = image.to(device).unsqueeze(0)  # Add batch dimension
                image.requires_grad = True
                
                # Forward pass the data through the model
                outputs = model(image)

                # Check if pred and target are the same to compute the loss or skip
                result = check_pred_target(outputs, target, True)
                if result is None:
                    continue

                # Calculate loss
                loss = loss_fn(outputs, target, device)
                model.zero_grad()
                loss.backward()

                # Generate adversarial example
                data_grad = image.grad.data
                adv_image = fgsm_attack(image, epsilon, data_grad)
                adv_outputs = model(adv_image)

                # Check if adv_pred and target are the same
                result = check_pred_target(adv_outputs, target, False)

                # If adv_pred and target differ, save the adversarial example
                if result is not None:
                    if i <= 400:
                        save_fgsm_image_label(output_dir, adv_image, target, folder, class_mapping)
                        i += 1
                else:
                    correct += 1

            # Calculate and log accuracy for this folder
            final_acc = correct / float(len(random_indices))
            print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(random_indices)} = {final_acc}")

            # Clear GPU memory after processing each folder
            del dataset
            torch.cuda.empty_cache()


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

    # Run test for each epsilon
    for eps in EPSILONS:
        create_adversarial_attack(eps, BASE_DIR, OUTPUT_DIR, WIDTH, HEIGHT, CLASSES, CLASS_MAPPING, device)


    print('Generation adversarial attack has completed successfully')


if __name__ == '__main__':
    args = parse_opt()
    main(args)