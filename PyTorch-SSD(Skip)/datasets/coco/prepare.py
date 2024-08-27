import os
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO


def save_as_json(basename, dataset):
    filename = os.path.join(os.path.dirname(__file__), basename)
    print("Saving %s ..." % filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def process_split(split_dir):
    annotation_file = os.path.join(split_dir, '_annotations.coco.json')
    coco = COCO(annotation_file)

    dataset = []
    ids = sorted(coco.imgs.keys())
    for id in tqdm(ids):
        image_info = coco.loadImgs(id)[0]
        image_path = os.path.join(split_dir, image_info["file_name"])
        anno = coco.loadAnns(coco.getAnnIds(id))
        boxes, classes = [], []
        for obj in anno:
            if obj['iscrowd'] == 0:
                xmin, ymin, w, h = obj['bbox']
                if w <= 0 or h <= 0:
                    print(f"Skip object with degenerate bbox (w={w:.2f}, h={h:.2f}).")
                    continue
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                classes.append(coco.getCatIds().index(obj['category_id']-1))
        dataset.append({
            'image': os.path.abspath(image_path),
            'boxes': boxes,
            'classes': classes,
            'difficulties': [0 for _ in classes]
        })

    return dataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help="Path to the root directory containing 'train', 'test', and 'valid' folders."
    )
    args = parser.parse_args()

    for split in ['train', 'test', 'valid']:
        split_dir = os.path.join(args.root, split)
        dataset = process_split(split_dir)
        save_as_json(f"{split}.json", dataset)


if __name__ == '__main__':
    main()
