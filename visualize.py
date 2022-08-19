import os
import json
import numpy as np
import torch
torch.cuda.current_device()
from torch.utils.data import DataLoader

from datasets import *
from BiCross.Visualizer import Visualizer


with open('visualization_config.json', 'r') as f:
    config = json.load(f)
config_json = json.dumps(config, sort_keys=True, indent=4, separators=(',', ':'))
print(config_json)

np.random.seed(config['general']['seed'])

if config['general']['dataset'] == "vkitti":
    test_dataset = VKITTI(
        dataset_path = "dataset_aaai/vkitti/2.0.3",
        mode='test',
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        drop_last=False
    )
elif config['general']['dataset'] == "kitti":
    test_dataset = KITTI(
        dataset_path = "dataset_aaai/kitti",
        mode='test',
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "rkitti":
    test_dataset = RKITTI(
        dataset_path = "dataset_aaai/rkitti",
        mode='test',
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "ekitti":
    test_dataset = EKITTI(
        dataset_path = "dataset_aaai/ekitti",
        mode='test',
        testId=39,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "drivingstereo":
    test_dataset = DrivingStereo(
        dataset_path = "dataset_aaai/drivingstereo",
        mode='test',
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "ministereo":
    test_dataset = MiniStereo(
        dataset_path = "dataset_aaai/ministereo",
        mode='test',
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "weather":
    test_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='test',
        weather=config['general']['weather'],
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif "nyu" in config['general']['dataset']:
    test_nyu_dataset = NYU(
        dataset_path = "dataset_aaai/nyu/v2",
        mode='test',
    )
    test_nyu_dataloader = DataLoader(
        test_nyu_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif "respike" in config['general']['dataset']:
    test_respike_dataset = Respike(
        dataset_path = "dataset_aaai/respike",
        mode='test',
        scene='indoor',
    )
    test_respike_dataloader = DataLoader(
        test_respike_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )

visualizer = Visualizer(config)
visualizer.visualize(test_dataloader)
