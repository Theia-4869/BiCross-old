import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class DrivingStereoWeather(Dataset):
    def __init__(self, dataset_path, mode, size="half", weather="all"):
        self.mode = mode
        self.size = size
        self.weather = weather

        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid 'mode' parameter: {}".format(mode))
        if mode == "val":
            mode = "test"

        if size not in ["full", "half"]:
            raise ValueError("Invalid 'size' parameter: {}".format(size))
        
        data_infos = []
        for w in os.listdir(dataset_path):
            if weather != "all" and weather != w:
                continue

            weather_path = os.path.join(dataset_path, w, "input", size)
            filenames = os.listdir(os.path.join(weather_path, "depth"))
            filenames.sort()
            filenum = len(filenames)
            if mode == "train":
                filenames = filenames[:int(filenum*0.8)]
            else:
                filenames = filenames[int(filenum*0.8):]
            for filename in filenames:
                depth_filename = os.path.join(weather_path, "depth", filename)
                rgb_filename = os.path.join(weather_path, "rgb", filename)
                spike_filename = os.path.join(weather_path, "spike/npy", filename.replace("png", "npy"))

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
        left = int((w - 864) / 2)
        top = int(h - 384)
        data_item["rgb"] = data_item["rgb"].crop((left, top, left+864, top+384))

        _, h, w = data_item["spike"].shape
        left = int((w - 864) / 2)
        top = int(h - 384)
        data_item["spike"] = data_item["spike"][:, top:top + 384, left:left + 864]

        if self.mode == "train":
            h, w = data_item["depth"].shape
            left = int((w - 864) / 2)
            top = int(h - 384)
            data_item["depth"] = data_item["depth"][top:top + 384, left:left + 864]

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