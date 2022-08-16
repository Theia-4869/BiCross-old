import os
import copy
import torch
torch.cuda.current_device()
import numpy as np
import wandb

from os import replace
from tqdm import tqdm
from cv2 import cv2
from numpy.core.numeric import Inf
from timm. models. layers import StdConv2dSame
from BiCross.utils import get_losses, get_optimizer, get_schedulers, create_dir
from DPT.models import DPTDepthModel

class Visualizer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config['general']['device'] if torch.cuda.is_available() else "cpu")
        
        if config['general']['model_type'] == 'vit_base':
            if config['general']['modality'] == 'spike':
                self.model = DPTDepthModel(
                    path=config['general']['path_spike_checkpoint'],
                    scale=0.00006016,
                    shift=0.00579,
                    invert=True,
                    backbone='vitb16_384',
                    non_negative=False,
                    enable_attention_hooks=False,
                    with_uncertainty=True,
                    spike_input=True,
                )
            else:
                self.model = DPTDepthModel(
                path=config['general']['path_spike_checkpoint'],
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone='vitb16_384',
                non_negative=False,
                enable_attention_hooks=False,
                with_uncertainty=True,
            )

        elif config['general']['model_type'] == 'hybrid':
            if config['general']['modality'] == 'spike':
                self.model = DPTDepthModel(
                    path=config['general']['path_spike_checkpoint'],
                    scale=0.00006016,
                    shift=0.00579,
                    invert=True,
                    backbone="vitb_rn50_384",
                    non_negative=False,
                    enable_attention_hooks=False,
                    with_uncertainty=True,
                    spike_input=True,
                )
            else:
                self.model = DPTDepthModel(
                    path=config['general']['path_rgb_checkpoint'],
                    scale=0.00006016,
                    shift=0.00579,
                    invert=True,
                    backbone="vitb_rn50_384",
                    non_negative=False,
                    enable_attention_hooks=False,
                    with_uncertainty=True,
                )

        else:
            raise ValueError("wrong model type: ", config['general']['model_type'])

        self.model.to(self.device)

        # depth loss
        self.loss_depth, self.loss_uncertainty, self.loss_consistency = get_losses(config)

    def visualize(self, test_dataloader):
        if self.config['wandb']['enable']:
            wandb.init(
                project="BiCross-vis",
                name=self.config['wandb']['name'],
                entity=self.config['wandb']['username'],
            )

        new_val_loss, abs_rel = self.run_eval(test_dataloader)

        print('Finished Testing')

    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        self.model.eval()

        val_loss = 0.
        rgb_1 = None
        spike_1 = None
        depth_1 = None
        output_1 = None
        uncertainty_1 = None
        path_1_1 = None
        path_2_1 = None
        path_3_1 = None
        path_4_1 = None
        errors = {"abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0,"a3":0}

        length = len(val_dataloader)
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Visualizing ...")

            for i, sample in enumerate(pbar):
                rgb, spike, depth = sample['rgb'],  sample['spike'], sample['depth']
                rgb, spike, depth = rgb.to(self.device), spike.to(self.device), depth.to(self.device)
                
                if self.config['general']['modality'] == 'rgb':
                    output, uncertainty, path_1, path_2, path_3, path_4, layer_4, cls_token = self.model(rgb)
                elif self.config['general']['modality'] == 'spike':
                    output, uncertainty, path_1, path_2, path_3, path_4, layer_4, cls_token = self.model(spike)

                output = output.unsqueeze(1)
                uncertainty = uncertainty.unsqueeze(1)
                if "kitti" in self.config['general']['dataset']:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[352, 1216],
                        mode="bicubic",
                        align_corners=False,
                    )
                    uncertainty = torch.nn.functional.interpolate(
                        uncertainty,
                        size=[352, 1216],
                        mode="bicubic",
                        align_corners=False,
                    )
                else:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[384, 864],
                        mode="bicubic",
                        align_corners=False,
                    )
                    uncertainty = torch.nn.functional.interpolate(
                        uncertainty,
                        size=[384, 864],
                        mode="bicubic",
                        align_corners=False,
                    )
                output = output.squeeze(1)
                uncertainty = uncertainty.squeeze(1)

                depth = depth.squeeze(1) #1xHxW -> HxW
                _, height, width = depth.shape

                if "kitti" in self.config['general']['dataset']:
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 352, left_margin:left_margin + 1216] = output
                else:
                    top_margin = int(height - 384)
                    left_margin = int((width - 864) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 384, left_margin:left_margin + 864] = output
                output = output_uncropped

                output[output < 1e-3] = 1e-3
                output[output > 80] = 80
                output[torch.isinf(output)] = 80

                depth[torch.isinf(depth)] = 0
                depth[torch.isnan(depth)] = 0

                valid_mask = torch.logical_and(depth > 1e-3, depth < 80)
                eval_mask = torch.zeros(valid_mask.shape).to(self.device)
                eval_mask[:, int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)

                rgb_1 = rgb
                spike_1 = spike
                depth_1 = depth
                output_1 = output
                uncertainty_1 = uncertainty
                path_1_1 = path_1[:, 0, :, :]
                path_2_1 = path_2[:, 0, :, :]
                path_3_1 = path_3[:, 0, :, :]
                path_4_1 = path_4[:, 0, :, :]
                self.img_logger(rgb_1, spike_1, depth_1, output_1, uncertainty_1, path_1_1, path_2_1, path_3_1, path_4_1)

                # get loss
                loss = self.loss_depth(output[valid_mask], depth[valid_mask])
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss' : val_loss/(i+1)})

                # calculate metric
                output = np.array(output[valid_mask].detach().cpu(), dtype = np.float32)#.squeeze(0)
                depth = np.array(depth[valid_mask].detach().cpu(), dtype = np.float32)#.squeeze(0)

                abs_rel, sq_rel, rmse, rmse_log, a1, a2,a3 = self.validate_errors(depth, output)
                errors["abs_rel"] = errors["abs_rel"] + abs_rel
                errors["rmse"] = errors["rmse"] + rmse
                
                errors["sq_rel"] = errors["sq_rel"] + sq_rel
                errors["rmse_log"] = errors["rmse_log"] + rmse_log
                errors["a1"] = errors["a1"] + a1
                errors["a2"] = errors["a2"] + a2
                errors["a3"] = errors["a3"] + a3
                
            errors["abs_rel"] = errors["abs_rel"] / length
            errors["rmse"] = errors["rmse"] / length  
            errors["sq_rel"] = errors["sq_rel"] / length
            errors["rmse_log"] = errors["rmse_log"] / length
            errors["a1"] = errors["a1"] / length
            errors["a2"] = errors["a2"] / length
            errors["a3"] = errors["a3"] / length
            
            abs_rel = errors["abs_rel"]
            sq_rel = errors["sq_rel"]
            rmse = errors["rmse"]
            rmse_log = errors["rmse_log"]
            a1 = errors["a1"]
            a2 = errors["a2"]
            a3 = errors["a3"]

            print("errors evaluate disparity:\n abs_rel: {}, rmse: {}, sq_rel: {}, rmse_log: {}, a1: {}, a2: {}, a3: {}".format(
                abs_rel, rmse, sq_rel, rmse_log, a1, a2,a3 ))

            if self.config['wandb']['enable']:
                wandb.log({"val_abs_rel": abs_rel})
                wandb.log({"val_sq_rel": sq_rel})
                wandb.log({"val_rmse": rmse})
                wandb.log({"val_rmse_log": rmse_log})
                wandb.log({"val_a1": a1})
                wandb.log({"val_a2": a2})
                wandb.log({"val_a3": a3})
                wandb.log({"val_loss": val_loss/(i+1)})
        
        return val_loss/(i+1), abs_rel

    def img_logger(self, rgb, spike, depth, output, uncertainty, path_1, path_2, path_3, path_4):
        nb_to_show = self.config['wandb']['images_to_show'] if self.config['wandb']['images_to_show'] <= len(rgb) else len(rgb)

        tmp = rgb[:nb_to_show].detach().cpu().numpy()
        rgbs = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = np.repeat(np.mean(spike[:nb_to_show][:, 0:16, :, :].unsqueeze(2).detach().cpu().numpy(), axis=1), 3, axis=1)
        spikes = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = depth[:nb_to_show].detach().cpu().numpy()
        depths = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = output[:nb_to_show].detach().cpu().numpy()
        outputs = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = torch.abs(output - depth) / depth
        tmp[tmp > 1.0] = 1.0
        tmp = np.repeat(tmp[:nb_to_show].unsqueeze(1).detach().cpu().numpy(), 3, axis=1)
        errors = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = np.repeat(uncertainty[:nb_to_show].unsqueeze(1).detach().cpu().numpy(), 3, axis=1)
        uncertainties = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        tmp = path_1[:nb_to_show].detach().cpu().numpy()
        paths_1 = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        # tmp = path_2[:nb_to_show].detach().cpu().numpy()
        # paths_2 = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        # tmp = path_3[:nb_to_show].detach().cpu().numpy()
        # paths_3 = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())

        # tmp = path_4[:nb_to_show].detach().cpu().numpy()
        # paths_4 = (tmp - tmp.min() + 1e-8) / (tmp.max() - tmp.min())
        
        rgbs = rgbs.transpose(0,2,3,1)
        spikes = spikes.transpose(0,2,3,1)

        # depth = (depth-0.3)*2
        # depth[depth < 0.0] = 0.0
        # depth[depth > 1.0] = 1.0

        # output = (output-0.3)*2
        # output[output < 0.0] = 0.0
        # output[output > 1.0] = 1.0

        depths = np.uint8(depths * 255)
        outputs = np.uint8(outputs * 255)

        errors = errors.transpose(0,2,3,1)
        uncertainties = uncertainties.transpose(0,2,3,1)
        paths_1 = np.uint8(paths_1 * 255)
        # paths_2 = np.uint8(paths_2 * 255)
        # paths_3 = np.uint8(paths_3 * 255)
        # paths_4 = np.uint8(paths_4 * 255)

        output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))
        wandb.log({
            "rgb": [wandb.Image(cv2.resize(im, output_dim), caption='rgb_{}'.format(i+1)) for i, im in enumerate(rgbs)],
            "spike": [wandb.Image(cv2.resize(im, output_dim), caption='spike_{}'.format(i+1)) for i, im in enumerate(spikes)],
            "depth": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_RAINBOW), caption='depth_{}'.format(i+1)) for i, im in enumerate(depths)],
            "output": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_RAINBOW), caption='output_{}'.format(i+1)) for i, im in enumerate(outputs)],
            "error": [wandb.Image(cv2.resize(im, output_dim), caption='error_{}'.format(i+1)) for i, im in enumerate(errors)],
            "uncertainty": [wandb.Image(cv2.resize(im, output_dim), caption='uncertainty_{}'.format(i+1)) for i, im in enumerate(uncertainties)],
            "feature": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_JET), caption='feature_{}'.format(i+1)) for i, im in enumerate(paths_1)],
            # "path_2": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_JET), caption='path_2_{}'.format(i+1)) for i, im in enumerate(paths_2)],
            # "path_3": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_JET), caption='path_3_{}'.format(i+1)) for i, im in enumerate(paths_3)],
            # "path_4": [wandb.Image(cv2.applyColorMap(cv2.resize(im, output_dim), cv2.COLORMAP_JET), caption='path_4_{}'.format(i+1)) for i, im in enumerate(paths_4)],
        })

    def validate_errors(self, gt, pred): # for disparity
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
