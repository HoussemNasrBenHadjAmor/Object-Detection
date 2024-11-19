import torch
from torchvision.transforms import functional as Ft
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
from ultralytics import YOLO
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
import numpy as np
import os
from tqdm import tqdm
from utils.general import (TQDM_BAR_FORMAT)
import uuid
import yaml
import argparse
import cv2
import random

from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

from utils.loss_tal_dual import ComputeLoss


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


def main(args):
    # Initialize W&B with project name.
    # wandb_init(name=args['project_name'])
    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Settings/parameters/constants.
    MODEL_PATH = data_configs['MODEL_PATH']
    DATA_TRAINING_PATH = data_configs['DATA_TRAINING_PATH']
    BASE_DIR = data_configs['BASE_DIR']
    OUTPUT_DIR = data_configs['OUTPUT_DIR']
    EPSILONS = data_configs['EPSILONS']  
    WIDTH = data_configs['WIDTH']    
    HEIGHT = data_configs['HEIGHT']
    SAVE_DIR_EXAMPLES_PATH = data_configs['SAVE_DIR_EXAMPLES_PATH']
    IMGSZ = data_configs['IMGSZ']
    BATCH_SIZE = data_configs['BATCH_SIZE']
    
    use_cuda=True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False, data=DATA_TRAINING_PATH, fp16=False)
    model.eval()

    def denormalize_bboxes(target, image_width, image_height, device):
        denormalized_target = []
        for bbox in target:
            # bbox = [class_id, x_center, y_center, width, height]
            x_center, y_center, bbox_width, bbox_height = bbox[1:]

            # Denormalize the center coordinates
            x_center = x_center * image_width
            y_center = y_center * image_height

            # Denormalize the width and height
            bbox_width = bbox_width * image_width
            bbox_height = bbox_height * image_height

            # Calculate the corner coordinates
            x_min = x_center - (bbox_width / 2)
            y_min = y_center - (bbox_height / 2)
            x_max = x_center + (bbox_width / 2)
            y_max = y_center + (bbox_height / 2)

            denormalized_target.append([x_min, y_min, x_max, y_max])

        return torch.tensor(denormalized_target, dtype=torch.float32).to(device)

    # YOLO loss function
    def loss_fn(outputs, targets, device, lambda_coord=5, lambda_noobj=0.5):
        # Extract predicted boxes, objectness scores, and class probabilities
    
        predicted_bboxes = outputs[0][:, :4]
        predicted_obj_scores = outputs[0][:,4]
        predicted_class_probs = outputs[0][:,5]
    
        target_bboxes = denormalize_bboxes(targets[:,1:], 640, 640, device)
        target_obj_scores = torch.ones(targets.size(0), device=device)
        target_class_probs = targets[:,1].to(device)
    
        # Bounding Box Loss (MSE Loss)
        bbox_loss = F.smooth_l1_loss(predicted_bboxes, target_bboxes)

    
        # Objectness Loss (BCE Loss)
        obj_loss = F.binary_cross_entropy_with_logits(predicted_obj_scores, target_obj_scores)
    

        # Class Probability Loss (BCE Loss)
        class_loss = F.binary_cross_entropy_with_logits(predicted_class_probs, target_class_probs)
    
        total_loss = bbox_loss + obj_loss + class_loss
        return total_loss

    # FGSM function to generate adversarial examples
    def fgsm_attack(image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def check_pred_target(output, target, skip):
        output_shape = output[0].shape[0]
        target_shape = len(target)
        if output_shape != target_shape:
            if skip:
                return None
            else:
                return ''   
        else:
            if skip:
                return ''
            else:
                return None


    def save_fgsm_image_label(output_dir, image, target, folder):
        tensor = image.cpu().detach()
        # Remove the batch dimension and convert to NumPy array
        image = tensor.squeeze().numpy()
        # Transpose dimensions from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        # Convert values to [0, 255] for uint8 images
        image = (np.clip(image * 255, 0, 255)).astype(np.uint8)

        
        image_dir = os.path.join(output_dir, folder, 'images')
        label_dir = os.path.join(output_dir, folder, 'labels')

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    
        # Generate a random image_name using UUID    
        random_filename = str(uuid.uuid4())
        image_name = random_filename + '.jpg'
        path_name = random_filename + '.txt'
        fsgm_image_path = os.path.join(image_dir, image_name)    
        fsgm_label_path = os.path.join(label_dir, path_name)    

        cv2.imwrite(fsgm_image_path, image)

        target_data = target[:, -5:].clone()
        target_data[:, 0] = target[:, 1].int()

        extracted_data_list = target_data.tolist()

        # Save to a .txt file with the desired formatting
        with open(fsgm_label_path, 'w') as f:
            for row in extracted_data_list:
                formatted_row = f'{int(row[0])} ' + ' '.join(f'{value:.5f}' for value in row[1:])
                f.write(formatted_row + '\n')


    def create_adversarial_attack(epsilon, base_dir, imgsz, batch_size, output_dir):
        # Process the entire validation dataset
        adv_examples = []
        correct = 0

        for folder  in ['train', 'valid', 'test']:
            image_dir = os.path.join(base_dir, folder)
            _, dataset = create_dataloader(image_dir , imgsz, batch_size, 32)

            for img, target, _, _ in tqdm(dataset, desc=f'Processing adversarial attack in {folder} images'):
                im = img.float().to(device)
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]

                im.requires_grad = True    
    
                outputs = model(im)[0]  # Get the raw predictions from the model

                pred = non_max_suppression(outputs) # Get the prediction from non_max_suppression
    
                # Check if pred and target are the same to compute the loss or just skip
                result = check_pred_target(pred, target, True)

                if result is None:
                    continue

                # Calculate loss
                loss = loss_fn(pred, target, device)  # Assuming loss function and model expect labels     

                model.zero_grad()
                loss.backward()
                data_grad = im.grad.data

                adv_image = fgsm_attack(im, epsilon, data_grad)
                adv_outputs = model(adv_image)

                # Check if adv_pred and target are the same
                result = check_pred_target(adv_outputs, target, False)

                if result is not None:
                    # Means the adv_pred and target are not the same
                    #adv_examples.append(adv_image)
                    save_fgsm_image_label(output_dir, adv_image, target, folder)
                else:
                    # Means the adv_pred and labels are the same
                    correct += 1

            # Calculate final accuracy for this epsilon
            final_acc = correct / float(len(dataset))
            print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(dataset)} = {final_acc}")

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples            

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

    adv_examples = []
    # Run test for each epsilon
    for eps in EPSILONS:
        _, ex = create_adversarial_attack(eps, BASE_DIR, IMGSZ, BATCH_SIZE, OUTPUT_DIR)
        #adv_examples.extend(ex)

    # Select 10 random examples
    #random_examples = random.sample(adv_examples, 10 if len(adv_examples) > 10 else len(adv_examples))

    #save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

    print('Generation adversarial attack has completed successfully')


if __name__ == '__main__':
    args = parse_opt()
    main(args)
