import os
import json
import numpy as np
import torch
torch.cuda.current_device()
from torch.utils.data import DataLoader

from datasets import *
from BiCross.Tester import Tester


with open('test_config.json', 'r') as f:
    config = json.load(f)
config_json = json.dumps(config, sort_keys=True, indent=4, separators=(',', ':'))
print(config_json)

np.random.seed(config['general']['seed'])

if config['general']['dataset'] == "vkitti":
    test_vkitti_dataset = VKITTI(
        dataset_path = "/home/notebook/data/group/liujiaming/spike/dataset_aaai/vkitti/2.0.3",
        mode='test',
    )
    test_vkitti_dataloader = DataLoader(
        test_vkitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        drop_last=False
    )
elif config['general']['dataset'] == "kitti":
    test_kitti_dataset = KITTI(
        dataset_path = "/home/notebook/data/group/liujiaming/spike/dataset_aaai/kitti",
        mode='test',
    )
    test_kitti_dataloader = DataLoader(
        test_kitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "ekitti":
    test_ekitti_dataset = EKITTI(
        dataset_path = "/home/notebook/data/group/liujiaming/spike/dataset_aaai/ekitti",
        mode='test',
        testId=39,
    )
    test_ekitti_dataloader = DataLoader(
        test_ekitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "drivingstereo":
    test_drivingstereo_dataset = DrivingStereo(
        dataset_path = "/home/notebook/data/group/liujiaming/spike/dataset_aaai/drivingstereo",
        mode='test',
    )
    test_drivingstereo_dataloader = DataLoader(
        test_drivingstereo_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
elif config['general']['dataset'] == "weather":
    test_weather_dataset = DrivingStereoWeather(
        dataset_path = "/home/notebook/data/group/liujiaming/spike/dataset_aaai/drivingstereoweather",
        mode='test',
        weather=config['general']['weather'],
    )
    test_weather_dataloader = DataLoader(
        test_weather_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )

tester = Tester(config)

if config['general']['dataset'] == "vkitti":
    tester.test(test_vkitti_dataloader)
elif config['general']['dataset'] == "kitti":
    tester.test(test_kitti_dataloader)
elif config['general']['dataset'] == "ekitti":
    tester.test(test_ekitti_dataloader)
elif config['general']['dataset'] == "drivingstereo":
    tester.test(test_drivingstereo_dataloader)
elif config['general']['dataset'] == "weather":
    tester.test(test_weather_dataloader)
