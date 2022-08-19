import os
import torch
torch.cuda.current_device()
import numpy as np
import wandb

from tqdm import tqdm
from itertools import cycle
from cv2 import cv2
from numpy.core.numeric import Inf
from timm. models. layers import StdConv2dSame

from BiCross.utils import get_losses, get_optimizer, get_schedulers, create_dir
from DPT.models import DPTDepthModel
from DPT.functions import ReverseLayerF

class DomainCrosser(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config['general']['device'] if torch.cuda.is_available() else "cpu")
        self.alpha = config['general']['alpha']
        self.dmax = config['general']['dmax']

        spike_input = False
        if config['general']['modality'] == 'spike':
            spike_input = True
        
        if config['general']['stage'] == 'crossdomain':
            if config['general']['model_type'] == 'hybrid':
                if config['general']['scene'] == 'outdoor':
                    self.model_student = DPTDepthModel(
                        path=config['general']['path_spike_pretrain'],
                        scale=0.00006016,
                        shift=0.00579,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        enable_attention_hooks=False,
                        with_uncertainty=True,
                        spike_input=spike_input,
                    )
                    self.model_teacher = DPTDepthModel(
                        path=config['general']['path_spike_pretrain'],
                        scale=0.00006016,
                        shift=0.00579,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        enable_attention_hooks=False,
                        with_uncertainty=True,
                        spike_input=spike_input,
                    )
                elif config['general']['scene'] == 'indoor':
                    self.model_student = DPTDepthModel(
                        path=config['general']['path_spike_pretrain'],
                        scale=0.000305,
                        shift=0.1378,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        enable_attention_hooks=False,
                        with_uncertainty=True,
                        spike_input=spike_input,
                    )
                    self.model_teacher = DPTDepthModel(
                        path=config['general']['path_spike_pretrain'],
                        scale=0.000305,
                        shift=0.1378,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=False,
                        enable_attention_hooks=False,
                        with_uncertainty=True,
                        spike_input=spike_input,
                    )

            else:
                raise ValueError("wrong model type: ", config['general']['model_type'])
            
        else:
            raise ValueError("wrong training stage: ", config['general']['stage'])

        self.head_source = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(1),
            torch.nn.Linear(768,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,2),
            torch.nn.LogSoftmax(dim=1),
        )
        self.head_target = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(1),
            torch.nn.Linear(768,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,2),
            torch.nn.LogSoftmax(dim=1),
        )

        for param in self.model_teacher.parameters():
            param.detach_()
        self.model_teacher.eval()
        
        self.model_student.to(self.device)
        self.model_teacher.to(self.device)
        self.head_source.to(self.device)
        self.head_target.to(self.device)

        # loss
        self.loss_depth, self.loss_uncertainty, self.loss_contrastive = get_losses(config)
        self.loss_adv = torch.nn.NLLLoss()

        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model_student)
        self.optimizer_source = torch.optim.Adam(self.head_source.parameters(), lr=config['general']['lr_backbone'])
        self.optimizer_target = torch.optim.Adam(self.head_target.parameters(), lr=config['general']['lr_backbone'])
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch, self.optimizer_source, self.optimizer_target])

    def train(self, dataloaders):
        source_dataloader, target_dataloader, test_dataloader = dataloaders
        epochs = self.config['general']['epochs']
        
        if self.config['wandb']['enable']:
            wandb.init(project="BiCross", entity=self.config['wandb']['username'])
            wandb.config = {
                "learning_rate_backbone": self.config['general']['lr_backbone'],
                "learning_rate_scratch": self.config['general']['lr_scratch'],
                "epochs": epochs,
                "batch_size": self.config['general']['batch_size']
            }
        
        val_loss = Inf
        best_abs_rel = 100
        best_epoch = 0
        global_step = 0

        if self.config['general']['warmup']:
            print("Wram up epoch")
            running_loss = 0.0
            self.model_student.train()

            pbar = tqdm(source_dataloader)
            pbar.set_description("Warming up ...")
            warmup_step = 0

            for batch_idx, sample in enumerate(pbar):
                warmup_step += 1
                if warmup_step >= self.config['general']['warmup_step']:
                    break
                spike, depth = sample['spike'], sample['depth']
                spike, depth = spike.to(self.device), depth.to(self.device)

                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()

                # training depth
                output, uncertainty, _, _, _, _, _, _ = self.model_student(spike)

                output = output.unsqueeze(1)
                uncertainty = uncertainty.unsqueeze(1)
                if "kitti" in self.config['general']['dataset'][0]:
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
                elif "nyu" in self.config['general']['dataset'][0]:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[480, 640],
                        mode="bicubic",
                        align_corners=False,
                    )
                    uncertainty = torch.nn.functional.interpolate(
                        uncertainty,
                        size=[480, 640],
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

                output[output < 1e-3] = 1e-3

                _, height, width = depth.shape
                valid_mask = torch.logical_and(depth > 1e-3, depth < self.dmax)
                eval_mask = torch.zeros(valid_mask.shape).to(self.device)
                eval_mask[:, int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)
                
                # get depth loss
                loss_depth = self.loss_depth(output[valid_mask], depth[valid_mask])

                # get uncertainty loss
                loss_uncertainty = self.loss_uncertainty(uncertainty[valid_mask], output[valid_mask], depth[valid_mask])

                # get all loss
                loss = 0.1 * (loss_depth + loss_uncertainty)

                loss.backward()

                # step optimizer
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                # log loss
                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        depth.min().item(), depth.max().item(),'\n',
                        output.min().item(), output.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)

                if self.config['wandb']['enable'] and ((batch_idx % 20 == 0 and batch_idx > 0) or batch_idx == len(train_dataloader) - 1):
                    wandb.log({"loss": running_loss/(batch_idx+1)})
                pbar.set_postfix({'training_loss': running_loss / (batch_idx+1)})

        for epoch in range(epochs):
            print("Epoch : ", epoch)
            running_loss = 0.0
            unc_thresh = 0.0
            self.model_student.train()

            if self.config['general']['method'] == 'adversarial':

                train_dataloader = zip(source_dataloader, target_dataloader)
                pbar = tqdm(train_dataloader, total=len(target_dataloader))
                pbar.set_description("Adversarial training ...")

                for batch_idx, (source_sample, target_sample) in enumerate(pbar):
                    source_spike, target_spike, source_depth = source_sample['spike'], target_sample['spike'], source_sample['depth']
                    source_spike, target_spike, source_depth = source_spike.to(self.device), target_spike.to(self.device), source_depth.to(self.device)

                    self.optimizer_backbone.zero_grad()
                    self.optimizer_scratch.zero_grad()
                    self.optimizer_source.zero_grad()
                    self.optimizer_target.zero_grad()

                    # training for source
                    output_source, uncertainty_source, _, _, _, _, layer_4_source, _ = self.model_student(source_spike)

                    # training for target
                    _, _, _, _, _, _, layer_4_target, _ = self.model_student(target_spike)

                    output_source = output_source.unsqueeze(1)
                    uncertainty_source = uncertainty_source.unsqueeze(1)
                    if "kitti" in self.config['general']['dataset'][0]:
                        output_source = torch.nn.functional.interpolate(
                            output_source,
                            size=[352, 1216],
                            mode="bicubic",
                            align_corners=False,
                        )
                        uncertainty_source = torch.nn.functional.interpolate(
                            uncertainty_source,
                            size=[352, 1216],
                            mode="bicubic",
                            align_corners=False,
                        )
                    elif "nyu" in self.config['general']['dataset'][0]:
                        output_source = torch.nn.functional.interpolate(
                            output_source,
                            size=[480, 640],
                            mode="bicubic",
                            align_corners=False,
                        )
                        uncertainty_source = torch.nn.functional.interpolate(
                            uncertainty_source,
                            size=[480, 640],
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        output_source = torch.nn.functional.interpolate(
                            output_source,
                            size=[384, 864],
                            mode="bicubic",
                            align_corners=False,
                        )
                        uncertainty_source = torch.nn.functional.interpolate(
                            uncertainty_source,
                            size=[384, 864],
                            mode="bicubic",
                            align_corners=False,
                        )
                    output_source = output_source.squeeze(1)
                    uncertainty_source = uncertainty_source.squeeze(1)
                    output_source[output_source < 1e-3] = 1e-3

                    _, height, width = source_depth.shape
                    valid_mask = torch.logical_and(source_depth > 1e-3, source_depth < self.dmax)
                    eval_mask = torch.zeros(valid_mask.shape).to(self.device)
                    eval_mask[:, int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
                    valid_mask = torch.logical_and(valid_mask, eval_mask)
                    
                    # get depth loss
                    loss_depth = self.loss_depth(output_source[valid_mask], source_depth[valid_mask])

                    # get uncertainty loss
                    loss_uncertainty = self.loss_uncertainty(uncertainty_source[valid_mask], output_source[valid_mask], source_depth[valid_mask])

                    # get adversarial loss
                    beta = self.config['general']['beta']
                    reverse_layer_4_source = ReverseLayerF.apply(layer_4_source, beta)
                    reverse_layer_4_target = ReverseLayerF.apply(layer_4_target, beta)

                    source_output = self.head_source(reverse_layer_4_source)
                    source_label = torch.zeros(self.config['general']['batch_size']).long().cuda()
                    loss_adv_source = self.loss_adv(source_output, source_label)

                    target_output = self.head_target(reverse_layer_4_target)
                    target_label = torch.ones(self.config['general']['batch_size']).long().cuda()
                    loss_adv_target = self.loss_adv(target_output, target_label)

                    # get all loss
                    loss = loss_depth + loss_uncertainty + 0.1 * (loss_adv_source + loss_adv_target)

                    loss.backward()

                    # step optimizer
                    self.optimizer_scratch.step()
                    self.optimizer_backbone.step()
                    self.optimizer_source.step()
                    self.optimizer_target.step()

                    # log loss
                    running_loss += loss.item()
                    if np.isnan(running_loss):
                        print('\n',
                            layer_4_source.min().item(), layer_4_source.max().item(),'\n',
                            layer_4_target.min().item(), layer_4_target.max().item(),'\n',
                            loss.item(),
                        )
                        exit(0)

                    if self.config['wandb']['enable'] and ((batch_idx % 20 == 0 and batch_idx > 0) or batch_idx == len(train_dataloader) - 1):
                        wandb.log({"loss": running_loss/(batch_idx+1)})
                    pbar.set_postfix({'training_loss': running_loss / (batch_idx+1)})

            elif self.config['general']['method'] == 'meanteacher':

                pbar = tqdm(target_dataloader)
                pbar.set_description("Mean Teaching ...")

                for batch_idx, sample in enumerate(pbar):
                    spike_weak, spike_strong = sample['spike'], sample['spike']
                    spike_weak, spike_strong = spike_weak.to(self.device), spike_strong.to(self.device)

                    self.optimizer_backbone.zero_grad()
                    self.optimizer_scratch.zero_grad()

                    output_depths_pseudo, output_uncertainty_pseudo, _, _, _, _, _, _ = self.model_teacher(spike_weak)
                    output_depths, output_uncertainty, _, _, _, _, _, _ = self.model_student(spike_strong)

                    _, height, width = output_depths_pseudo.shape
                    valid_mask = torch.logical_and(output_depths_pseudo > 1e-3, output_depths_pseudo < self.dmax)
                    eval_mask = torch.zeros(valid_mask.shape).to(self.device)
                    eval_mask[:, int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
                    valid_mask = torch.logical_and(valid_mask, eval_mask)
                    output_uncertainty_pseudo_sorted, _ = torch.sort(output_uncertainty_pseudo[valid_mask])
                    uncertainty_idx = int(output_uncertainty_pseudo_sorted.shape[0] * (0.5))
                    uncertainty_thresh = output_uncertainty_pseudo_sorted[uncertainty_idx]
                    uncertainty_mask = output_uncertainty_pseudo < uncertainty_thresh + 1e-8
                    valid_mask = torch.logical_and(valid_mask, uncertainty_mask)

                    output_depths[output_depths < 1e-3] = 1e-3
                    output_uncertainty_pseudo[output_uncertainty_pseudo < 0.0] = 0.0
                    output_uncertainty_pseudo[output_uncertainty_pseudo > 1.0] = 1.0

                    # get consistency loss
                    loss_consistency = self.loss_depth(output_depths[valid_mask], output_depths_pseudo[valid_mask])
                    
                    # get all loss
                    loss = loss_consistency

                    loss.backward()

                    # step optimizer
                    self.optimizer_scratch.step()
                    self.optimizer_backbone.step()

                    # ema update
                    for ema_param, param in zip(self.model_teacher.parameters(), self.model_student.parameters()):
                        ema_param.data.mul_(self.alpha).add_(param.data, alpha=(1 - self.alpha))

                    # log loss
                    running_loss += loss.item()
                    unc_thresh += uncertainty_thresh.item()
                    if np.isnan(running_loss):
                        print('\n',
                            output_depths_pseudo.min().item(), output_depths_pseudo.max().item(),'\n',
                            output_depths.min().item(), output_depths.max().item(),'\n',
                            loss.item(),
                        )
                        exit(0)

                    if self.config['wandb']['enable'] and ((batch_idx % 20 == 0 and batch_idx > 0) or batch_idx == len(train_drivingstereo_unlabeled_dataloader) - 1):
                        wandb.log({'loss': running_loss/(batch_idx+1)})
                    pbar.set_postfix({'loss': running_loss / (batch_idx+1), 'unc': unc_thresh / (batch_idx+1)})

            new_val_loss, abs_rel = self.run_eval(test_dataloader)

            if self.config['general']['save_model']:
                if new_val_loss < val_loss:
                    self.save_model(epoch)
                    val_loss = new_val_loss
                if abs_rel < best_abs_rel:
                    self.save_model(epoch, best=True)
                    best_abs_rel = abs_rel
                    best_epoch = epoch
                if epoch == epochs-1:
                    self.save_model(epoch, last=True)
            print("best_abs_rel: {}, best_epoch: {}".format(best_abs_rel, best_epoch))

            for scheduler in self.schedulers:
                scheduler.step(new_val_loss)

        print('Finished Training')

    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        self.model_student.eval()

        val_loss = 0.
        rgb_1 = None
        depth_1 = None
        output_1 = None
        errors = {"abs_rel":0, "sq_rel":0, "rmse":0, "rmse_log":0, "a1":0, "a2":0,"a3":0}

        length = len(val_dataloader)
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation ...")

            for i, sample in enumerate(pbar):
                spike, depth = sample['spike'], sample['depth']
                spike, depth = spike.to(self.device), depth.to(self.device)

                output, _, _, _, _, _, _, _ = self.model_student(spike)

                output = output.unsqueeze(1)
                if "kitti" in self.config['general']['dataset'][-1]:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[352, 1216],
                        mode="bicubic",
                        align_corners=False,
                    )
                elif "nyu" in self.config['general']['dataset'][-1]:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[480, 640],
                        mode="bicubic",
                        align_corners=False,
                    )
                elif "respike" in self.config['general']['dataset'][-1]:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=[250, 400],
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
                output = output.squeeze(1)

                depth = depth.squeeze(1) #1xHxW -> HxW
                _, height, width = depth.shape

                if "kitti" in self.config['general']['dataset'][-1]:
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 352, left_margin:left_margin + 1216] = output
                elif "nyu" in self.config['general']['dataset'][-1]:
                    top_margin = int(height - 480)
                    left_margin = int((width - 640) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 480, left_margin:left_margin + 640] = output
                elif "respike" in self.config['general']['dataset'][-1]:
                    top_margin = int(height - 250)
                    left_margin = int((width - 400) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 250, left_margin:left_margin + 400] = output
                else:
                    top_margin = int(height - 384)
                    left_margin = int((width - 864) / 2)
                    output_uncropped = torch.zeros_like(depth, dtype=torch.float32)
                    output_uncropped[:, top_margin:top_margin + 384, left_margin:left_margin + 864] = output
                output = output_uncropped

                output[output < 1e-3] = 1e-3
                output[output > self.dmax] = self.dmax
                output[torch.isinf(output)] = self.dmax

                depth[torch.isinf(depth)] = 0
                depth[torch.isnan(depth)] = 0

                valid_mask = torch.logical_and(depth > 1e-3, depth < self.dmax)
                eval_mask = torch.zeros(valid_mask.shape).to(self.device)
                eval_mask[:, int(0.40810811 * height):int(0.99189189 * height), int(0.03594771 * width):int(0.96405229 * width)] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)

                if i==0:
                    depth_1 = depth
                    output_1 = output

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
                wandb.log({"val_rmse": rmse})
                wandb.log({"val_a1": a1})
                # wandb.log({"val_a2": a2})
                # wandb.log({"val_a3": a3})
                wandb.log({"val_loss": val_loss/(i+1)})
                self.img_logger(rgb_1, depth_1, output_1, data_type=data_type)
        
        return val_loss/(i+1), abs_rel

    def save_model(self, epoch, best=False, last=False):
        if best == True:
            path_model = os.path.join(self.config['general']['path_model'], self.model_student.__class__.__name__)
            create_dir(path_model)
            torch.save({'model_state_dict': self.model_student.state_dict(),
                        'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                        'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                        }, path_model+'_best.pt')
            print('best Model saved at : {}'.format(path_model))
        elif last ==True:
            path_model = os.path.join(self.config['general']['path_model'], self.model_student.__class__.__name__)
            create_dir(path_model)
            torch.save({'model_state_dict': self.model_student.state_dict(),
                        'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                        'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                        }, path_model+'_last.pt')
            print('latest val loss Model saved at : {}'.format(path_model))
        else:
            path_model = os.path.join(self.config['general']['path_model'], self.model_student.__class__.__name__)
            create_dir(path_model)
            torch.save({'model_state_dict': self.model_student.state_dict(),
                        'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                        'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                        }, path_model+'_lowest.pt')
            print('lowest val loss Model saved at : {}'.format(path_model))

    def img_logger(self, rgb, depth, output, data_type=""):
        nb_to_show = self.config['wandb']['images_to_show'] if self.config['wandb']['images_to_show'] <= len(X) else len(X)
        tmp = X[:nb_to_show].detach().cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if output != None:
            tmp = depth[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_truths = np.repeat(tmp, 3, axis=1) / tmp.max()
            depth_preds = output[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_preds = np.repeat(depth_preds, 3, axis=1) / depth_preds.max()
        
        imgs = imgs.transpose(0,2,3,1)
        if output != None:
            depth_truths = depth_truths.transpose(0,2,3,1)
            depth_preds = depth_preds.transpose(0,2,3,1)
        
        output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))
        # wandb.log({
        #     "img": [wandb.Image(cv2.resize(im, output_dim), caption='img_{}'.format(i+1)) for i, im in enumerate(imgs)]
        # })
        if output != None:
            wandb.log({
                "depth_truths": [wandb.Image(cv2.resize(im, output_dim), caption='depth_truths_{}'.format(i+1)) for i, im in enumerate(depth_truths)],
                "depth_preds": [wandb.Image(cv2.resize(im, output_dim), caption='depth_preds_{}'.format(i+1)) for i, im in enumerate(depth_preds)]
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
