import os
import json
import shutil
import mmcv
from pycocotools.coco import COCO


def ensure_dir(path, clean=False):
    if os.path.exists(path) and clean:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_json(source_json_fp):
    with open(source_json_fp) as f:
        json_obj = json.load(f)
    return json_obj


def write_json(json_obj, target_json_fp):
    with open(target_json_fp, 'w') as f:
        json.dump(json_obj, f)


def merge_coco_json(json_fp1, json_fp2, target_json_fp):
    """Merge json_fp2 to json_fp1
    """
    js1 = mmcv.load(str(json_fp1))
    js2 = mmcv.load(str(json_fp2))
    coco2 = COCO(json_fp2)

    max_img_idx = max([item['id'] for item in js1['images']])
    max_ann_idx = max([item['id'] for item in js1['annotations']])

    img_index = max_img_idx + 1
    ann_index = max_ann_idx + 1
    for img_item in js2['images']:
        img_item = img_item.copy()
        old_img_id = img_item['id']
        new_img_id = img_index
        img_item['id'] = new_img_id
        js1['images'].append(img_item)
        img_index += 1
        ann_ids = coco2.getAnnIds(imgIds=old_img_id)
        anns = coco2.loadAnns(ids=ann_ids).copy()
        for ann_item in anns:
            ann_item['id'] = ann_index
            ann_item['image_id'] = new_img_id
            js1['annotations'].append(ann_item)
            ann_index += 1
    mmcv.dump(js1, str(target_json_fp))
