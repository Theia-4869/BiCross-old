import os
import random
import copy
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class EKITTI(Dataset):
    def __init__(self, dataset_path, mode, testId=39):
        self.mode = mode

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid 'mode' parameter: {}".format(mode))
        if mode == "val":
            mode = "test"

        train_scenes = ["2011_09_26_drive_0035_sync", "2011_09_26_drive_0046_sync", "2011_09_26_drive_0064_sync", "2011_09_30_drive_0020_sync"]
        test_scenes = ["2011_09_26_drive_00{}_sync".format(testId)]    # 19 23 39
        
        data_infos = []
        for scene in os.listdir(os.path.join(dataset_path, "input")):
            if mode == "train":
                if scene not in train_scenes:
                    continue
            else:
                if scene not in test_scenes:
                    continue

            scene_path = os.path.join(dataset_path, "input", scene)
            for filename in os.listdir(os.path.join(scene_path, "depth")):
                depth_filename = os.path.join(scene_path, "depth", filename)
                rgb_filename = os.path.join(scene_path, "rgb", filename)
                spike_filename = os.path.join(scene_path, "spike/npy", filename.replace("png", "npy"))

                if os.path.exists(rgb_filename) and os.path.exists(spike_filename):
                    data_infos.append({
                        "depth": depth_filename,
                        "rgb": rgb_filename,
                        "spike": spike_filename,
                    })
        
        data_infos.sort(key=lambda e:e.__getitem__("depth"))    
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        item_files = self.data_infos[idx]
        data_item = self._read_data(item_files)

        w, h = data_item["rgb"].size
        left = int((w - 1216) / 2)
        top = int(h - 352)
        data_item["rgb"] = data_item["rgb"].crop((left, top, left+1216, top+352))

        _, h, w = data_item["spike"].shape
        left = int((w - 1216) / 2)
        top = int(h - 352)
        data_item["spike"] = data_item["spike"][:, top:top + 352, left:left + 1216]

        if self.mode == "train":
            h, w = data_item["depth"].shape
            left = int((w - 1216) / 2)
            top = int(h - 352)
            data_item["depth"] = data_item["depth"][top:top + 352, left:left + 1216]
        
        to_tensor = transforms.ToTensor()
        data_item["rgb"] = to_tensor(data_item["rgb"])
        data_item["spike"] = torch.FloatTensor(data_item["spike"])
        data_item["depth"] = torch.FloatTensor(data_item["depth"])

        scale_rgb = transforms.Resize([384, 384], InterpolationMode.BILINEAR)
        scale_spike = transforms.Resize([384, 384], InterpolationMode.NEAREST)
        normalize = transforms.Normalize([.5, .5, .5], [.5, .5, .5])

        data_item["rgb"] = scale_rgb(data_item["rgb"])
        data_item["spike"] = scale_spike(data_item["spike"])
        data_item["rgb"] = normalize(data_item["rgb"])

        # if self.mode == "train":
        #     data_item["spike_strong"] = copy.deepcopy(data_item["spike"])

        #     frames = data_item["spike"].split(16, dim=0)
        #     new_frames = []
        #     for f in frames:
        #         copies = list(f.split(1, dim=0))
        #         random.shuffle(copies)
        #         copies = torch.cat(copies, dim=0)
        #         new_frames.append(copies)
        #     data_item["spike_strong"] = torch.cat(new_frames, dim=0)

        #     sparse_mask = (torch.rand(data_item["spike_strong"].shape) > 0.9)
        #     data_item["spike_strong"][sparse_mask] = -1

        #     erase = transforms.RandomErasing(p=1, scale=(0.01,0.05), value=-1.0)
        #     data_item["spike_strong"] = erase(data_item["spike_strong"])

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

        depth = depth_png.astype(np.float32) / 256.
        return depth

    def _read_rgb(self, rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        return rgb
    
    def _read_spike(self, spike_path):
        spike_mat = np.load(spike_path).astype(np.float32)
        spike = 2 * spike_mat - 1
        return spike
