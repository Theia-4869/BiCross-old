import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class NYU(Dataset):
    def __init__(self, dataset_path, mode):
        self.mode = mode

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid 'mode' parameter: {}".format(mode))
        if mode == "val":
            mode = "test"

        split_file = open(os.path.join(dataset_path, "{}.txt".format(mode)), 'r')
        
        data_infos = []
        for line in split_file:
            rgbfile, depthfile = line.split(' ')
            if depthfile[-1] == '\n':
                depthfile = depthfile[:-1]

            tokens = rgbfile.split('/')
            scene, seq, filename = tokens[0], tokens[1], tokens[2]
            rgb_filename = os.path.join(dataset_path, "input", scene, seq, "rgb", filename)
            spike_filename = os.path.join(dataset_path, "input", scene, seq, "spike", filename.replace("r", "s").replace("jpg", "npy"))

            tokens = depthfile.split('/')
            scene, seq, filename = tokens[0], tokens[1], tokens[2]
            depth_filename = os.path.join(dataset_path, "input", scene, seq, "depth", filename)

            if os.path.exists(rgb_filename) and os.path.exists(spike_filename) and os.path.exists(depth_filename):
                data_infos.append({
                    "rgb": rgb_filename,
                    "spike": spike_filename,
                    "depth": depth_filename,
                })
        
        split_file.close()
        data_infos.sort(key=lambda e:e.__getitem__("depth"))    
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
        normalize = transforms.Normalize([.5, .5, .5], [.5, .5, .5])

        data_item["rgb"] = scale_rgb(data_item["rgb"])
        data_item["spike"] = scale_spike(data_item["spike"])
        data_item["rgb"] = normalize(data_item["rgb"])

        # print("depth: ", data_item["depth"].shape)
        # print("rgb: ", data_item["rgb"].shape)
        # print("spike: ", data_item["spike"].shape)

        return data_item
    
    def _read_data(self, item_files):
        data = {}
        data['depth'] = self._read_depth(item_files['depth'])
        data['rgb'] = self._read_rgb(item_files['rgb'])
        data['spike'] = self._read_spike(item_files['spike'])
        return data

    def _read_depth(self, depth_path):
        # (copy from kitti devkit)
        # loads depth map D from png file
        # and returns it as a numpy array,

        depth_png = np.array(Image.open(depth_path), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 1000.
        return depth

    def _read_rgb(self, rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        return rgb
    
    def _read_spike(self, spike_path):
        spike_mat = np.load(spike_path).astype(np.float32)
        spike_mat = spike_mat[0:128]
        spike = 2 * spike_mat - 1
        return spike
