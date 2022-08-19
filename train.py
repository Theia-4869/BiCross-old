import os
import json
import numpy as np
import torch
torch.cuda.current_device()
from torch.utils.data import DataLoader

from datasets import *
from BiCross import *


with open('train_config.json', 'r') as f:
    config = json.load(f)
config_json = json.dumps(config, sort_keys=True, indent=4, separators=(',', ':'))
print(config_json)

np.random.seed(config['general']['seed'])

if "vkitti" in config['general']['dataset']:
    train_vkitti_dataset = VKITTI(
        dataset_path = "dataset_aaai/vkitti/2.0.3",
        mode='train',
    )
    test_vkitti_dataset = VKITTI(
        dataset_path = "dataset_aaai/vkitti/2.0.3",
        mode='test',
    )

    train_vkitti_dataloader = DataLoader(
        train_vkitti_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_vkitti_dataloader = DataLoader(
        test_vkitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        drop_last=False
    )
if "kitti" in config['general']['dataset']:
    train_kitti_dataset = KITTI(
        dataset_path = "dataset_aaai/kitti",
        mode='train',
    )
    test_kitti_dataset = KITTI(
        dataset_path = "dataset_aaai/kitti",
        mode='test',
    )

    train_kitti_dataloader = DataLoader(
        train_kitti_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_kitti_dataloader = DataLoader(
        test_kitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "ekitti" in config['general']['dataset']:
    train_ekitti_dataset = EKITTI(
        dataset_path = "dataset_aaai/ekitti",
        mode='train',
    )
    test_ekitti_dataset = EKITTI(
        dataset_path = "dataset_aaai/ekitti",
        mode='test',
    )

    train_ekitti_dataloader = DataLoader(
        train_ekitti_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_ekitti_dataloader = DataLoader(
        test_ekitti_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "drivingstereo" in config['general']['dataset']:
    train_drivingstereo_dataset = DrivingStereo(
        dataset_path = "dataset_aaai/drivingstereo",
        mode='train',
    )
    test_drivingstereo_dataset = DrivingStereo(
        dataset_path = "dataset_aaai/drivingstereo",
        mode='test',
    )

    train_drivingstereo_dataloader = DataLoader(
        train_drivingstereo_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_drivingstereo_dataloader = DataLoader(
        test_drivingstereo_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "weather" in config['general']['dataset']:
    train_weather_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='train',
        weather=config['general']['weather'],
    )
    test_weather_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='test',
        weather=config['general']['weather'],
    )

    train_weather_dataloader = DataLoader(
        train_weather_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_weather_dataloader = DataLoader(
        test_weather_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "weathers" in config['general']['dataset']:
    train_source_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='train',
        weather=config['general']['weather'][0],
    )
    train_target_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='train',
        weather=config['general']['weather'][1],
    )
    test_target_dataset = DrivingStereoWeather(
        dataset_path = "dataset_aaai/drivingstereoweather",
        mode='test',
        weather=config['general']['weather'][1],
    )

    train_source_dataloader = DataLoader(
        train_source_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8,
        drop_last=True
    )
    train_target_dataloader = DataLoader(
        train_target_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8,
        drop_last=True
    )
    test_target_dataloader = DataLoader(
        test_target_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "nyu" in config['general']['dataset']:
    train_nyu_dataset = NYU(
        dataset_path = "dataset_aaai/nyu/v2",
        mode='train',
    )
    test_nyu_dataset = NYU(
        dataset_path = "dataset_aaai/nyu/v2",
        mode='test',
    )

    train_nyu_dataloader = DataLoader(
        train_nyu_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_nyu_dataloader = DataLoader(
        test_nyu_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )
if "respike" in config['general']['dataset']:
    train_respike_dataset = Respike(
        dataset_path = "dataset_aaai/respike",
        mode='train',
        scene='indoor',
    )
    test_respike_dataset = Respike(
        dataset_path = "dataset_aaai/respike",
        mode='test',
        scene='indoor',
    )

    train_respike_dataloader = DataLoader(
        train_respike_dataset, 
        batch_size=config['general']['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        drop_last=True
    )
    test_respike_dataloader = DataLoader(
        test_respike_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8,
        drop_last=False
    )

if config['general']['stage'] == 'pretrain':
    trainer = Pretrainer(config)
elif config['general']['stage'] == 'neuromorphic':
    trainer = NeuromorphicTrainer(config)
elif config['general']['stage'] == 'crossmodality':
    trainer = ModalityCrosser(config)
elif config['general']['stage'] == 'crossdomain':
    trainer = DomainCrosser(config)

if config['general']['stage'] == 'pretrain' or config['general']['stage'] == 'neuromorphic' or config['general']['stage'] == 'crossmodality':
    if "vkitti" in config['general']['dataset']:
        trainer.train((train_vkitti_dataloader, test_vkitti_dataloader))
    elif "kitti" in config['general']['dataset']:
        trainer.train((train_kitti_dataloader, test_kitti_dataloader))
    elif "ekitti" in config['general']['dataset']:
        trainer.train((train_ekitti_dataloader, test_ekitti_dataloader))
    elif "drivingstereo" in config['general']['dataset']:
        trainer.train((train_drivingstereo_dataloader, test_drivingstereo_dataloader))
    elif "weather" in config['general']['dataset']:
        trainer.train((train_weather_dataloader, test_weather_dataloader))
    elif "nyu" in config['general']['dataset']:
        trainer.train((train_nyu_dataloader, test_nyu_dataloader))
    elif "respike" in config['general']['dataset']:
        trainer.train((train_respike_dataloader, test_respike_dataloader))
elif config['general']['stage'] == 'crossdomain':
    if "vkitti" in config['general']['dataset'] and "ekitti" in config['general']['dataset']:
        trainer.train((train_vkitti_dataloader, train_ekitti_dataloader, test_ekitti_dataloader))
    elif "vkitti" in config['general']['dataset'] and "kitti" in config['general']['dataset']:
        trainer.train((train_vkitti_dataloader, train_kitti_dataloader, test_kitti_dataloader))
    elif "kitti" in config['general']['dataset'] and "drivingstereo" in config['general']['dataset']:
        trainer.train((train_kitti_dataloader, train_drivingstereo_dataloader, test_drivingstereo_dataloader))
    elif "weathers" in config['general']['dataset']:
        trainer.train((train_source_dataloader, train_target_dataloader, test_target_dataloader))
    elif "nyu" in config['general']['dataset'] and "respike" in config['general']['dataset']:
        trainer.train((train_nyu_dataloader, train_respike_dataloader, test_respike_dataloader))
