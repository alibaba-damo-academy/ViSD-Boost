import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from monai import transforms
from monai.data.utils import dense_patch_slices
from typing import Any, Callable, List, Sequence, Tuple, Union


def masks_to_boxes_3d(masks):
    """Compute the bounding boxes around the provided 3D masks

    The masks should be in format [N, D, H, W] where N is the number of masks, (D, H, W) are the spatial dimensions.

    Returns a [N, 6] tensor, with the boxes in min_x, min_y, min_z, max_x, max_y, max_z format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)

    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)

def collate_fn(batch):
    return batch[0]

@torch.no_grad()
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.

        """
        if len(image_size) != num_spatial_dims:
            raise ValueError("image coord different from spatial dims.")
        if len(roi_size) != num_spatial_dims:
            raise ValueError("roi coord different from spatial dims.")

        scan_interval = []
        for i in range(num_spatial_dims):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - overlap))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

def center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    
    crop_d, crop_h, crop_w = max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min), max(crop_size[2], x_max - x_min)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    
    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    
    return image[..., z_start:z_end, y_start:y_end, x_start:x_end], mask[..., z_start:z_end, y_start:y_end, x_start:x_end]

class DataFolder(Dataset):
    def __init__(self):
        super().__init__()

        vis_root = 'dir_path/processed_valid_images'

        img_paths = []
        for root, _, files in os.walk(vis_root):
            # for debug
            # if len(img_paths) == 6:
            #     break
            for file in files:
                if file.endswith('.nii.gz'):
                    img_paths.append(os.path.join(root, file))
        self.img_paths = img_paths

        self.organs = [
            'lung', 'heart', 'esophagus', 'aorta'
        ]

        self.loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        ])
        
        self.test_items = [
            ['lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'],
            ['lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'], 
            ['lung', 'Lung nodule', 'Not Lung nodule.', 'Lung nodule.'],
            ['lung', 'Lung opacity', 'Not Lung opacity.', 'Lung opacity.'],
            ['lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'],
            ['lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'],
            ['lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'],
            ['lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'],
            ['lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'],
            ['lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'],
            ['lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'],
            ['heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'],
            ['heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'],
            ['heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'],
            ['esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'],
            ['aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'],
        ]

        self.test_items = [tuple(item) for item in self.test_items]
        self.test_organs = list(set([item[0] for item in self.test_items]))

    @staticmethod
    def get_patient_id(image_path):
        img_name = image_path.split('/')[-1]
        patient_id = img_name.split('_')[0]
        return patient_id
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = image_path.replace('images', 'masks')
        
        file_name = image_path.split('/')[-1]
        input_path = {'image': image_path, 'label': mask_path}

        data = self.loader(input_path)
        
        test_organ_names = self.organs

        test_items = [test_item for test_item in self.test_items if test_item[0] in test_organ_names]
        
        meta_info = {
            'file_name': file_name,
            'img_path': image_path,
            'test_organ_names': test_organ_names
        }
        
        return data['image'].as_tensor(), data['label'].as_tensor(), test_items, meta_info

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--csv_file', type=str, help='The path to the CSV file for processing.')

    parser.add_argument("--cfg-path", required=False, default='infer_configs.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

@torch.inference_mode()
def evaluate():
    # def model
    from lavis.utils.vit import ViT  # pseudo vit
    from lavis.utils.model import VISDBOOST
    from lavis.utils.med import XBertEncoder
    
    image_encoder = ViT(
        in_channels=1,
        img_size=(112, 256, 352),
        patch_size=(16, 16, 32),
        num_classes=0,
        dropout_rate=0.1,
        qkv_bias=True
    ) 
    text_encoder = XBertEncoder.from_config(None, from_pretrained=True)

    model = VISDBOOST(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )
    
    # define dataloader
    datafolder = DataFolder()
    dataloader = DataLoader(
        datafolder,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    pad_func = transforms.DivisiblePadd(
                    keys=["image", "label"], 
                    k=(16, 32, 32),
                    mode='constant', 
                    constant_values=0,
                    method="end"
            )
    
    # load ckpt
    ckpt1 = torch.load(
            '../checkpoint_vqvae.pth',
            map_location='cpu'
        )
    model.load_state_dict(ckpt1['model'], strict=False)
    
    ckpt_path = f'../checkpoint.pth'
    ckpt = torch.load(
        ckpt_path, map_location='cpu'
    )
    model.load_state_dict(ckpt['model'], strict=False)

    model.eval()
    model.cuda()

    sw_batch_size = 4
    overlap = 0.25
    roi_size = (112, 256, 352)

    results = []
    text_feat_dict = model.prepare_text_feat(datafolder.test_items)
    organ_feat_dict = {}
    save_path = 'ZeroShotResult_rate.csv'

    for i, (image, mask, test_items, meta_info) in enumerate(tqdm(dataloader, desc='Infer')):
        fid = meta_info['file_name']
        organ_feat_dict[fid] = {}

        image = image[None].cuda()
        mask = mask[None].cuda()

        test_organs = meta_info['test_organ_names']
        
        whole_organ_sizes = dict(zip(test_organs, [torch.eq(mask, datafolder.organs.index(test_organ) + 1).sum().item() for test_organ in test_organs]))    

        test_organs = [test_organ for test_organ in test_organs if whole_organ_sizes[test_organ] > 0]
        test_items = [test_item for test_item in test_items if test_item[0] in test_organs]

        image_size = list(image.shape[2:])
        num_spatial_dims = len(image.shape) - 2

        scan_interval = _get_scan_interval(
            image_size, roi_size, num_spatial_dims, overlap
        )
        slices = dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)

        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        for k, v in organ_logits.items():
            if not len(v):
                organ_name = k[0]
                organ_id = datafolder.organs.index(organ_name)

                window_patch, window_mask = center_crop(
                    image,
                    torch.eq(mask, organ_id + 1),
                    crop_size=roi_size
                )

                window_mask = window_mask.float()
                window_mask[window_mask == 1] = organ_id + 1

                pad_data = pad_func({'image': window_patch[0], 'label': window_mask[0]})
                window_patch, window_mask = pad_data['image'], pad_data['label']

                organ_logits = model.forward_test_win(
                    window_patch[None], 
                    window_mask[None],
                    organ_logits,
                    test_organs,
                    text_feat_dict,
                    organ_feat_dict[fid],
                    whole_organ_sizes,
                    skip_organ=organ_id
                )
            
        res = [meta_info['file_name']] + [''] * len(datafolder.test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        for item, probs in organ_logits.items():
            res[datafolder.test_items.index(item) + 1] = np.concatenate(probs).mean(0)[1]
        results.append(res)
    
    pd.DataFrame(
        results,
        columns=['file_name'] + [f'_'.join(k) for k in datafolder.test_items]
    ).to_csv(save_path, index=False, encoding='utf-8')
    
    print('Save csv file successfully!')

if __name__ == '__main__':
    evaluate()
    


