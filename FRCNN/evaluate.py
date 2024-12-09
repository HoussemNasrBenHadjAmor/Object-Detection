"""
USAGE

# Training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --no-mosaic --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4
"""

from torch_utils.engine import (evaluate, calculate_precision, calculate_recall)

from utils.metrics import (ap_per_class)

from datasets import (
 create_valid_dataset, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, save_map50, save_map50_95, save_mAP, save_precisionB, save_recallB
)
from utils.logging import (
    set_log, 
    coco_log
)

import torch
import argparse
import yaml
import numpy as np
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# For same annotation colors each time.
np.random.seed(42)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', default=5, type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch-size', dest='batch_size', default=4, type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-ims', '--img-size', dest='img_size', default=640, type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-pn', '--project-name', default=None, type=str, dest='project_name',
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', dest='vis_transformed', action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '-nm', '--no-mosaic', dest='no_mosaic', action='store_false',
        help='pass this to not to use mosaic augmentation'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', dest='use_train_aug', action='store_true',
        help='whether to use train augmentation, uses some advanced augmentation \
              that may make training difficult when used with mosaic'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', dest='cosine_annealing', action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', default=None, type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', dest='resume_training', action='store_true',
        help='whether to resume training, if true, \
             loads previous training plots and epochs \
             and also loads the otpimizer state dictionary'
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
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = args['batch_size']
    OUT_DIR = set_training_dir(args['project_name'])
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    # Set logging file.
    set_log(OUT_DIR)
    # writer = set_summary_writer(OUT_DIR)

    # Model configurations
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
  
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )

    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # iterations till ena and plot graphs for all iterations.
    val_map_05 = []
    val_map = []
    val_precision = []
    val_recall = []
    val_mAP50 = []
    val_mAP50_95 = []
    val_precision_per_epoch = []
    val_recall_per_epoch = []


    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn']
        model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['config']['NC']
            CLASSES = checkpoint['config']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()


    coco_evaluator, stats, val_pred_image, category_ids, category_names, tp, conf, pred_cls, target_cls, fn_count = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        save_valid_preds=SAVE_VALID_PREDICTIONS,
        out_dir=OUT_DIR,
        classes=CLASSES,
        colors=COLORS
    )

    precision = calculate_precision(tp, pred_cls)
    recall = calculate_recall(tp, fn_count)

    # Append the precision & recall values
    val_precision_per_epoch.append(precision)
    val_recall_per_epoch.append(recall)

    # Append curent epoch's average loss to `train_loss_list_epoch`.
    print(f'stats : {stats}')
    val_map_05.append(stats[1])
    val_map.append(stats[0])
    val_precision.append(stats[0])
    val_recall.append(stats[8])
    val_mAP50.append(stats[1])
    val_mAP50_95.append(stats[0])
      
    np_tp = np.array(tp, dtype=bool)
    np_tp = np_tp[:, np.newaxis]
    ap_per_class(category_names,np_tp, np.array(conf), np.array(pred_cls), np.array(target_cls), plot=True , save_dir = OUT_DIR)
        
    # Save mAP50  for each epoch
    save_map50(val_mAP50, OUT_DIR, title='Metrics-mAP50(B)')
    # Save mAP50_95  for each epoch
    save_map50_95(val_mAP50_95, OUT_DIR, title='Metrics-mAP50-95(B)')
    save_precisionB(val_precision_per_epoch, OUT_DIR, title='Metrics-precision(B)')
    save_recallB(val_recall_per_epoch , OUT_DIR, title = 'Metrics-recall(B)')

    # Save mAP plots.
    save_mAP(OUT_DIR, val_map_05, val_map)

    coco_log(OUT_DIR, stats)

if __name__ == '__main__':
    args = parse_opt()
    main(args)