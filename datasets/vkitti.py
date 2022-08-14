import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class VKITTI(Dataset):
    def __init__(self, dataset_path, mode):
        self.mode = mode

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid 'mode' parameter: {}".format(mode))
        if mode == "val":
            mode = "test"

        train_scenes = []
        test_scenes = []
        if "1.3.1" in dataset_path:
            train_scenes = ["0001", "0002", "0006", "0020"]
            test_scenes = ["0018"]
        elif "2.0.3" in dataset_path:
            train_scenes = ["Scene01", "Scene02", "Scene06", "Scene20"]
            test_scenes = ["Scene18"]
        
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
                if "1.3.1" in dataset_path:
                    depth_filename = os.path.join(scene_path, "depth", filename)
                    rgb_filename = os.path.join(scene_path, "rgb", filename)
                    spike_filename = os.path.join(scene_path, "spike/npy", filename.replace("png", "npy"))
                elif "2.0.3" in dataset_path:
                    depth_filename = os.path.join(scene_path, "depth", filename)
                    rgb_filename = os.path.join(scene_path, "rgb", filename.replace("depth", "rgb").replace("png", "jpg"))
                    spike_filename = os.path.join(scene_path, "spike/npy", filename.replace("depth", "spike").replace("png", "npy"))

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

        # depth_png = np.array(Image.open(depth_path), dtype=int)
        depth_png = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 100.
        return depth

    def _read_rgb(self, rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        return rgb
    
    def _read_spike(self, spike_path):
        spike_mat = np.load(spike_path).astype(np.float32)
        spike = 2 * spike_mat - 1
        return spike
