import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
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
    
    use_cuda=True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # Load the state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Get the classes and classes number from the checkpoint
    NUM_CLASSES = checkpoint['config']['NC']
    #CLASSES = checkpoint['config']['CLASSES']

    # Create our model
    model = fasterrcnn_resnet50_fpn_v2(num_classes = NUM_CLASSES, pretrained=False, coco_model=False)

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

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
        image_name = random_filename + '.jpg'
        label_name = random_filename + '.xml'
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
    
    def create_adversarial_attack (epsilon, base_dir, output_dir, width, height, classes, class_mapping, device) :
    # Process the entire validation dataset
        adv_examples = []
        correct = 0
        i = 0
        for folder in ['train', 'valid', 'test']:
            image_dir = os.path.join(base_dir, folder)
            label_dir = os.path.join(base_dir, folder)
            dataset = create_valid_dataset(image_dir, label_dir, width, height, classes)
    
            for image , target in tqdm(dataset, desc=f'Processing adversarial attack in {folder} images'):
                image = image.to(device)
                image = image.unsqueeze(0)
            
                # Set requires_grad attribute of tensor. Important for Attack
                image.requires_grad = True
                # Forward pass the data through the model
                outputs = model(image)

                # Check if pred and target are the same to compute the loss or just skip
                result = check_pred_target(outputs , target, True)
                if result == None:
                    continue

                # Calculate loss
                loss = loss_fn(outputs, target, device)  # Assuming loss function and model expect labels

                model.zero_grad()
                loss.backward()

                data_grad = image.grad.data
                adv_image = fgsm_attack(image, epsilon, data_grad)
                adv_outputs = model(adv_image)

                # Check if adv_pred and target are the same
                result = check_pred_target(adv_outputs , target, False)

                if result != None:
                    # Means the adv_pred and target are not the same
                    adv_examples.append(adv_image)
                    save_fgsm_image_label(output_dir, adv_image, target, folder, class_mapping)
                else :
                    # Means the adv_pred and labels are the same
                    correct +=1

            # Calculate final accuracy for this epsilon
            final_acc = correct/float(len(dataset))
            # Increment i to save the images and targets depending on which folder --> exp train/test/valid
            i += 1
    
            print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(dataset)} = {final_acc}")

            # Clear GPU memory after processing each folder
            del dataset
            del image
            del target
            torch.cuda.empty_cache()


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
        _, ex = create_adversarial_attack(eps, BASE_DIR, OUTPUT_DIR, WIDTH, HEIGHT, CLASSES, CLASS_MAPPING, device)
        adv_examples.extend(ex)

    # Select 10 random examples
    random_examples = random.sample(adv_examples, 10 if len(adv_examples) > 10 else len(adv_examples))

    save_some_examples(random_examples, SAVE_DIR_EXAMPLES_PATH)

    print('Generation adversarial attack has completed successfully')


if __name__ == '__main__':
    args = parse_opt()
    main(args)