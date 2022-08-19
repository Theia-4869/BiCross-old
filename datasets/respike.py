import os
import re
import sys
import torch
import numpy as np
from PIL import Image
from cv2 import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class Respike(Dataset):
    def __init__(self, dataset_path, mode, scene="outdoor"):
        self.mode = mode
        self.scene = scene

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid 'mode' parameter: {}".format(mode))
        if mode == "test":
            mode = "val"

        if scene not in ["indoor", "outdoor"]:
            raise ValueError("Invalid 'scene' parameter: {}".format(scene))

        data_infos = []
        dataset_path = os.path.join(dataset_path, scene, "spike", mode)
        for scene in os.listdir(dataset_path):
            scene_path = os.path.join(dataset_path, scene, "left")
            for filename in os.listdir(scene_path):
                spike_filename = os.path.join(scene_path, filename)
                rgb_filename = os.path.join(scene_path[::-1].replace("spike"[::-1], "reconstruction"[::-1], 1)[::-1], filename.replace("npy", "png"))
                depth_filename = os.path.join(scene_path[::-1].replace("spike"[::-1], "depth_trans"[::-1], 1)[::-1].replace("left", ""), filename.replace("npy", "pfm"))

                if os.path.exists(rgb_filename) and os.path.exists(depth_filename):
                    data_infos.append({
                        "spike": spike_filename,
                        "rgb": rgb_filename,
                        "depth": depth_filename,
                    })
        
        data_infos.sort(key=lambda e:e.__getitem__("spike"))    
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        item_files = self.data_infos[idx]
        data_item = self._read_data(item_files)

        to_tensor = transforms.ToTensor()
        data_item["rgb"] = to_tensor(data_item["rgb"])
        data_item["spike"] = torch.FloatTensor(data_item["spike"])
        data_item["depth"] = torch.FloatTensor(data_item["depth"])

        scale_rgb = transforms.Resize([224, 224], InterpolationMode.BILINEAR)
        scale_spike = transforms.Resize([224, 224], InterpolationMode.NEAREST)

        data_item["rgb"] = scale_rgb(data_item["rgb"])
        data_item["spike"] = scale_spike(data_item["spike"])

        # print("rgb: ", data_item["rgb"].shape)
        # print("spike: ", data_item["spike"].shape)
        # print("depth: ", data_item["depth"].shape)

        return data_item
    
    def _read_data(self, item_files):
        data = {}
        data['rgb'] = self._read_rgb(item_files['rgb'])
        data['spike'] = self._read_spike(item_files['spike'])
        data['depth'] = self._read_depth(item_files['depth'])
        return data

    def _read_rgb(self, rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        return rgb
    
    def _read_spike(self, spike_path):
        spike_mat = np.load(spike_path).astype(np.float32)
        spike_mat = spike_mat[0:128]
        spike = 2 * spike_mat - 1

        spike = spike.transpose(1, 2, 0)
        spike = cv2.flip(spike, 0)
        for i in range(400 // 8):
            spike_part = spike[:, 8*i:8*(i+1), :]
            spike_part = cv2.flip(spike_part, 1)
            spike[:, 8*i:8*(i+1), :] = spike_part
        spike = spike.transpose(2, 0, 1)
        
        return spike

    def _read_depth(self, depth_path):
        with open(depth_path, 'rb') as depth_file:
            header = depth_file.readline().rstrip()
            if (sys.version[0]) == '3':
                header = header.decode('utf-8')
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise Exception('Not a PFM file.')

            if (sys.version[0]) == '3':
                dim_match = re.match(r'^(\d+)\s(\d+)\s$', depth_file.readline().decode('utf-8'))
            else:
                dim_match = re.match(r'^(\d+)\s(\d+)\s$', depth_file.readline())
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header.')

            if (sys.version[0]) == '3':
                scale = float(depth_file.readline().rstrip().decode('utf-8'))
            else:
                scale = float(depth_file.readline().rstrip())
                
            if scale < 0: # little-endian
                endian = '<'
                scale = -scale
            else:
                endian = '>' # big-endian

            depth = np.fromfile(depth_file, endian + 'f')
            shape = (height, width, 3) if color else (height, width)

            depth = np.reshape(depth, shape)
            depth = np.flipud(depth)
            depth = np.array(depth)
                
            return depth
