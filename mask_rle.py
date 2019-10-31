import mmcv
import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle(mask):
    rle = []
    if len(mask.shape)<=3:
        return mask_utils.encode(np.asfortranarray(mask))
    else:
        for one_mask in mask:
            rle.append(mask_utils.encode(np.asfortranarray(one_mask)))
    return rle


def rle_to_mask(rle):
    if isinstance(rle, dict) or mmcv.is_list_of(rle, dict):
        return mask_utils.decode(rle)
    assert mmcv.is_list_of(rle, list)
    masks = []
    for one_rle in rle:
        masks.append(mask_utils.decode(one_rle))
    return np.stack(masks)
