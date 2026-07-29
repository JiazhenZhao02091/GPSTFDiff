from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from src.logger import FusionLogger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil
import cv2
from diffusers import DDIMScheduler, DDPMScheduler
import tqdm
from ema_pytorch import EMA

import torch.nn as nn
import lpips
import pytorch_msssim

class Trainer:
    def __init__(
        self,
        congfig_path,
        train_root_dir_path,
        train_dir_prefix_list=[
            'checkpoints',
            'txt_logs',
            'backend_logs',
            'imgs',
            'configs',
        ],
        #
        txt_logger_name='fusion',
        txt_logger_level='INFO',
        #
        train_dataloader=None,
        val_dataloader=None,
        #
        model=None,
        device_info=None,
        #
        optimizer=None,
        scheduler=None,
        #
        metric_list=None,
        # 
        loss_fun=None,
        #
        predition_type="sample",
        # 
        is_warm_up = False,
        is_ema = False,
    ):
        self.init_loss_fun(loss_fun)

        self.init_train_params()
        self.init_train_dir(train_root_dir_path, train_dir_prefix_list, congfig_path)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader)
        self.init_model(model, is_ema)
        self.init_optimizer(optimizer)
        self.init_scheduler(scheduler)
        self.init_metric(metric_list)

        self.init_tracker()
        self.init_diff_schduler(predition_type)

        
    "HuggingFace DDIMScheduler"
    def init_diff_schduler(self, predition_type):
        self.predition_type = predition_type
        self.diffScheduler = DDIMScheduler(
            # prediction_type = "epsilon",
            prediction_type = predition_type,
            beta_schedule = "linear", # scaled_linear
        )

    def init_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun  # l1 or mse or ...
        # if loss_fun == "perc":
        #     print("use perc loss....")
        self.init_percetptual_loss(0.08)

    def init_percetptual_loss(self, lambda_percep):
        class PerceptualAdaptor(nn.Module):
            def __init__(self, in_channels=6, out_channels=3, backbone='vgg'):
                super().__init__()
                self.adapter = nn.Sequential(
                    nn.Conv2d(in_channels, 64 , kernel_size=1),
                    nn.Conv2d(64, out_channels , kernel_size=1)
                )
                self.lpips = lpips.LPIPS(net=backbone).eval()
                for p in self.lpips.parameters():
                    p.requires_grad = False  # 固定LPIPS特征提取器

            def forward(self, pred, target):
                # 通道映射
                pred_3 = torch.clamp(self.adapter(pred), -1, 1)
                target_3 = torch.clamp(self.adapter(target), -1, 1)
                # LPIPS距离（输出为 [B,1,1,1]）
                loss = self.lpips(pred_3, target_3).mean()  #TODO：多尺度 lpips 特征
                return loss
        self.lambda_percep = lambda_percep
        self.perc_losser = PerceptualAdaptor()

    def init_train_params(self):
        self.current_epoch = 0
        self.max_epoch = 5000
        self.val_interal = 200
        self.current_val_step = 0
        self.current_train_step = 0

    def init_train_dir(self, train_root_dir_path, train_dir_prefix_list, congfig_path):
        self.train_root_dir_path = Path(train_root_dir_path)
        self.train_dir_prefix_list = train_dir_prefix_list
        self.mkdir()
        shutil.copy(congfig_path, self.train_configs_dir)

    def mkdir(self):
        for train_dir_prefix in self.train_dir_prefix_list:
            train_dir = self.train_root_dir_path / train_dir_prefix
            train_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f'train_{train_dir_prefix}_dir', train_dir)
    "txt logger 和 tensorboard logger"
    def init_logger(self, txt_logger_name, txt_logger_level):
        self.txt_logger = FusionLogger(
            logger_name=txt_logger_name,
            log_file=self.train_txt_logs_dir / 'log.log',
            log_level=txt_logger_level,
        )
        self.backend_logger = SummaryWriter(self.train_backend_logs_dir)
    "   从配置文件加载train_dataloader和val_dataloader  "
    def init_dataloader(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
    "   从配置文件中加载模型并转移到cuda中  "
    def init_model(self, model, is_ema):
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.is_ema = is_ema
        self.ema = EMA(self.model, beta=0.995, update_every=1).to(self.device) if is_ema else None
        if hasattr(self, 'perc_losser'):
            self.perc_losser.to(self.device)

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer(params=self.model.parameters())

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to(self.device)
        self.metric_list = metric_list

    def init_tracker(self):
        key_list = [self.loss_fun + '_loss'] + [metric.__name__ for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def train(self):
        while self.current_epoch < self.max_epoch:
            self.train_epoch()
            "验证 + 保存检查点"
            if (
                self.current_epoch + 1
            ) % self.val_interal == 0 or self.current_epoch == 0:
                self.val()
                self.save_checkpoint()
            self.current_epoch += 1
    "训练一个epoch"
    def train_epoch(self):
        self.model.train()
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            model_input_list, loss_gt, metrics_gt = self.before_train_iter(
                iter_idx, data_per_batch
            )
            self.train_iter(iter_idx, model_input_list, loss_gt, metrics_gt)

            self.current_train_step += 1
        self.ema.update() if self.is_ema else None
        msg = f'epoch: {self.current_epoch}'
        for key, value in self.train_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'train/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)
    "准备每个 iter 之前需要的内容"
    def before_train_iter(self, iter_idx, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        return model_input_list, loss_gt, metrics_gt

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01'].to('cuda')
        coarse_img_02 = data_per_batch['coarse_img_02'].to('cuda')
        fine_img_01 = data_per_batch['fine_img_01'].to('cuda')
        fine_img_02 = data_per_batch['fine_img_02'].to('cuda')
        model_input_list = [
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            fine_img_02,
        ]
        return model_input_list

    def get_loss_gt(self, data_per_batch):
        return data_per_batch['fine_img_02'].to(self.device)

    def get_metrics_gt(self, data_per_batch):
        return data_per_batch['fine_img_02'].to(self.device)
    "训练一个 iter"
    def train_iter(self, iter_idx, model_input_list, loss_gt, metrics_gt):
        self.optimizer.zero_grad()

        c1, c2, f1, f2 = model_input_list
        # print(f"判断lossgt 和 真实值得差异: {loss_gt == f2}")
        noise = torch.randn_like(f2)
        batch_size = f2.shape[0]

        timesteps = torch.randint(0, self.diffScheduler.num_train_timesteps, (batch_size,), device=f2.device)
        noisy_image = self.diffScheduler.add_noise(f2, noise, timesteps)
        outputs = self.model(noisy_image, timesteps, c1, c2, f1)
        #TODO Sample最好一次训练使用的是噪声？
        if self.loss_fun == 'mse':
            pixel_loss = F.mse_loss(outputs, loss_gt) # mse loss
        elif self.loss_fun == 'l1':
            pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
        elif self.loss_fun == 'perc':
            pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
            lpips_loss = self.perc_losser(outputs, loss_gt)
            pixel_loss = pixel_loss + self.lambda_percep * lpips_loss
        elif self.loss_fun == 'lpips_ssim':
            pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
            lpips_loss = self.perc_losser(outputs, loss_gt)
            ssim_loss = 1 - pytorch_msssim.ssim(outputs, loss_gt, data_range=1.0, size_average=True)
            pixel_loss = 0.8 * pixel_loss + 0.05 * lpips_loss + 0.2 * ssim_loss
        else:
            raise ValueError(f'loss_fun {self.loss_fun} not supported')
        # pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss   # 预测干净图像
        pixel_loss.backward()  # backward
        self.optimizer.step()
        msg = f'epoch: {self.current_epoch}, iter: {iter_idx}, loss: {pixel_loss.item():.4e}'
        self.train_tracker.update(self.loss_fun + '_loss', pixel_loss.item())

        self.txt_logger.info(msg)
    " 启动验证 "
    def val(self):
        self.ema.ema_model.eval() if self.is_ema else self.model.eval() # 模型切换到验证状态
        self.val_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            (
                model_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_val_iter(iter_idx, data_per_batch)
            self.val_iter(
                iter_idx,
                model_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )
            self.current_val_step += 1
        msg = f'val epoch: {self.current_epoch}'
        for key, value in self.val_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_val_iter(self, iter_idx, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        show_img_list = [
            data_per_batch['coarse_img_01'],
            data_per_batch['coarse_img_02'],
            data_per_batch['fine_img_01'],
            data_per_batch['fine_img_02'],
        ]
        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()

        return (
            model_input_list,
            loss_gt,
            metrics_gt,
            show_img_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def val_iter(
        self,
        iter_idx,
        model_input_list,
        loss_gt,
        metrics_gt,
        show_img_list,
        key,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        with torch.no_grad():
            c1, c2, f1, f2 = model_input_list
            noisy_image = torch.randn_like(f2)
            
            # self.diffScheduler.set_timesteps(self.diffScheduler.num_train_timesteps)
            self.diffScheduler.set_timesteps(100)
            batch_size = f2.shape[0]
            # timesteps = torch.randint(0, self.diffScheduler.num_train_timesteps, (batch_size,), device=f1.device)
            for i, t in enumerate(tqdm.tqdm(self.diffScheduler.timesteps)):
                # print(f"采样过程中的 i 和 t 分别为 :{i}, {t}, type为: {type(i)}, {type(t)}")  # 0 990 1 980 2 970
                # 为整个 batch 构造同一个时间步向量（形状 [B]）
                t_batch = torch.full((batch_size,), t, device=f2.device, dtype=torch.long)
                # 依据调度器缩放模型输入（DDIM 下通常为恒等，但保持调用更稳妥）
                model_input = self.diffScheduler.scale_model_input(noisy_image, t_batch)
                # 1. predict noise residual
                if self.is_ema:
                    model_output = self.ema.ema_model(model_input, t_batch, c1, c2, f1)
                else:
                    model_output = self.model(model_input, t_batch, c1, c2, f1)
                # 2. compute less noisy image and set x_t -> x_t-1
                noisy_image = self.diffScheduler.step(model_output, t, noisy_image).prev_sample
            "最终的图像"
            outputs = noisy_image  

            # print(f"####### DeBug #######")
            # print(f"outputs 的形状为: {outputs.shape}")  # [1, 6, 256, 256]
            # print(f"--------- 判断图像是否是归一化 ---------")
            # print(f"outputs 的最大值为: {outputs.max().item()}, 最小值为: {outputs.min().item()}") # 1 -1
            # print(f"loss_gt 的最大值为: {loss_gt.max().item()}, 最小值为: {loss_gt.min().item()}") # 1 -0.9995999932289124
            # print(f"metrics_gt 的最大值为: {metrics_gt.max().item()}, 最小值为: {metrics_gt.min().item()}") # 1 -0.99
            # print(f"metrics_gt == loss_gt is {metrics_gt == loss_gt}") # True
            # print(f"--------- --------- ---------")
            # print(f"####################")
            # TODO: 默认使用了sample
            if self.loss_fun == 'mse':
                pixel_loss = F.mse_loss(outputs, loss_gt) # mse loss
            elif self.loss_fun == 'l1':
                pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
            elif self.loss_fun == 'perc':
                pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
                lpips_loss = self.perc_losser(outputs, loss_gt) # lpips loss
                pixel_loss = pixel_loss + self.lambda_percep * lpips_loss    # final loss 
            elif self.loss_fun == 'lpips_ssim':
                pixel_loss = F.l1_loss(outputs, loss_gt) # l1 loss
                lpips_loss = self.perc_losser(outputs, loss_gt)
                ssim_loss = 1 - pytorch_msssim.ssim(outputs, loss_gt, data_range=1.0, size_average=True)
                pixel_loss = 0.8 * pixel_loss + 0.05 * lpips_loss + 0.2 * ssim_loss    
            else:
                raise ValueError(f'loss_fun {self.loss_fun} not supported')
            self.val_tracker.update(self.loss_fun + '_loss', pixel_loss.item())
            msg += f', loss: {pixel_loss.item():.4e}'
        
        for metric in self.metric_list:
            metric_value = metric(
                (outputs + 1.0) / 2.0,
                (metrics_gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.backend_logger.add_scalar(
                f'val_runinng/{metric_name}', metric_value.item(), self.current_val_step
            )
            self.val_tracker.update(metric_name, metric_value.item())

        self.txt_logger.info(msg)

        ## TODO
        "保存 tif 格式图像(img_save)以及png 格式的图像(img_show)"
        save_dir_prefix = f'{dataset_name}/{self.current_epoch}/save_img'
        save_dir_path = self.train_imgs_dir / save_dir_prefix
        save_name = key.split('-')[-1]
        save_name = save_name[:8] + '_save_img_' + save_name[9:] + '.tif'
        if 'STIL' in save_dir_prefix:
            save_name = key + '.tif'
        self.img_save(
            outputs,
            save_dir_path,
            save_name,
            normalize_scale,
            normalize_mode,
        )

        show_dir_prefix = f'{dataset_name}/{self.current_epoch}/show_img'
        show_dir_path = self.train_imgs_dir / show_dir_prefix
        show_name = key.split('-')[-1]
        show_name = show_name[:8] + '_show_img_' + show_name[9:] + '.png'
        if 'STIL' in save_dir_prefix:
            show_name = key + '.png'

        self.img_show(
            show_img_list,
            outputs,
            show_dir_path,
            show_name,
            normalize_mode,
        )

    def img_save(
        self,
        save_tensor,
        save_dir_path,
        save_name,
        normalize_scale,
        normalize_mode,
    ):
        save_img = save_tensor[0].cpu().numpy().transpose(1, 2, 0)
        if normalize_mode == 1:
            save_img = save_img * normalize_scale
        elif normalize_mode == 2:
            save_img = (save_img + 1.0) / 2.0 * normalize_scale
        save_img = np.clip(save_img, 0, normalize_scale)
        if normalize_scale == 255:
            save_img = save_img.astype(np.uint8)
        elif normalize_scale == 1:
            save_img = save_img.astype(np.float32)
        elif normalize_scale == 10000:
            save_img = save_img.astype(np.uint16)
        save_img_path = save_dir_path / save_name
        save_dir_path.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(save_img_path, save_img)

    def img_show(
        self,
        show_img_list,
        pred_tensor,
        show_dir_path,
        show_name,
        normalize_mode,
        img_interval=10,
    ):
        show_len = len(show_img_list)
        h_num, w_num = 3, show_len // 2
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * h_num + img_interval,
                (w + img_interval) * w_num + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        for h_index in range(h_num - 1):
            for w_index in range(w_num):
                show_sub_img = (
                    show_img_list[h_index * w_num + w_index][0]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                if normalize_mode == 1:
                    show_sub_img = show_sub_img * 255.0
                elif normalize_mode == 2:
                    show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
                if c == 6:
                    show_sub_img = show_sub_img[:, :, (3, 2, 1)]
                show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
                show_sub_img = cv2.resize(
                    show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST
                )
                show_img[
                    img_interval * (h_index + 1)
                    + h_index * h : img_interval * (h_index + 1)
                    + (h_index + 1) * h,
                    img_interval * (w_index + 1)
                    + w_index * w : img_interval * (w_index + 1)
                    + (w_index + 1) * w,
                    :,
                ] = show_sub_img

        show_sub_img = pred_tensor[0].cpu().numpy().transpose(1, 2, 0)
        if normalize_mode == 1:
            show_sub_img = show_sub_img * 255.0
        elif normalize_mode == 2:
            show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
        if c == 6:
            show_sub_img = show_sub_img[:, :, (3, 2, 1)]
        show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
        show_sub_img = cv2.resize(show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST)
        show_img[
            img_interval * h_num + h * (h_num - 1) : img_interval * h_num + h * h_num,
            img_interval : img_interval + w,
            :,
        ] = show_sub_img

        show_img_path = show_dir_path / show_name
        show_dir_path.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)

    def save_checkpoint(self):
        if self.is_ema:
            data = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ema': self.ema.state_dict(),
            }
            torch.save(
                data,
                self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
            )
        else:
            torch.save(
                self.model.state_dict(),
                self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
            )
