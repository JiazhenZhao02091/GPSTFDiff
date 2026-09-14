from pathlib import Path
import shutil
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.tensorboard import SummaryWriter
import cv2

from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.logger import FusionLogger, Tracker


class Trainer:
    def __init__(
        self,
        congfig_path,
        train_root_dir_path,
        train_dir_prefix_list=[
            "checkpoints",
            "txt_logs",
            "backend_logs",
            "imgs",
            "configs",
        ],
        txt_logger_name="fusion",
        txt_logger_level="INFO",
        train_dataloader=None,
        val_dataloader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        metric_list=None,
        gradient_accumulation_steps=4,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # -------------------------
        # 1) Accelerator MUST be first
        # -------------------------
        self.accelerator = Accelerator(
            # mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            split_batches=True,  # 如果你的 dataloader batch 很大需要拆分
            kwargs_handlers=[ddp_kwargs]
        )
        # -------------------------
        # 2) 基本参数 / 目录 / logger
        # -------------------------
        self.init_train_params()
        self.init_train_dir(train_root_dir_path, train_dir_prefix_list, congfig_path)
        self.accelerator.wait_for_everyone() # wait for dir creation
        self.init_logger(txt_logger_name, txt_logger_level)

        # -------------------------
        # 3) 保存用户输入到属性（但不要 .to(device)）
        # -------------------------
        self._orig_train_dataloader = train_dataloader
        self._orig_val_dataloader = val_dataloader
        self._orig_model = model
        self._orig_optimizer = optimizer(params=self._orig_model.parameters())
        self._orig_scheduler = scheduler(self._orig_optimizer) if scheduler is not None else None
        self._orig_metric_list = metric_list if metric_list is not None else []

        # -------------------------
        # 4) 初始化并 prepare（重要：不要提前 .to(device)）
        # -------------------------
        self.init_dataloader()
        self.init_model_and_optimizer_and_scheduler()
        self.init_metric()
        self.init_tracker()

        # -------------------------
        # 5) 现在创建 EMA（不要 prepare ema）
        #     使用 accelerator.unwrap_model() 以避免 DDP 包装带来的副作用
        # -------------------------
        # EMA expects an nn.Module (not DDP-wrapped) — use unwrap_model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # initialize EMA with unwrapped model (it will internally keep a shadow copy)
        self.ema = EMA(unwrapped_model, beta=0.995, update_every=1)

    def init_train_params(self):
        self.current_epoch = 0
        self.max_epoch = 10000
        self.val_interal = self.max_epoch
        self.current_val_step = 0
        self.current_train_step = 0
        self.save_interal = 50

    def init_train_dir(self, train_root_dir_path, train_dir_prefix_list, congfig_path):
        self.train_root_dir_path = Path(train_root_dir_path)
        self.train_dir_prefix_list = train_dir_prefix_list
        self.mkdir()
        # Only main process copies config to avoid races
        if self.accelerator.is_main_process:
            shutil.copy(congfig_path, self.train_configs_dir)

    def mkdir(self):
        for train_dir_prefix in self.train_dir_prefix_list:
            train_dir = self.train_root_dir_path / train_dir_prefix
            train_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f"train_{train_dir_prefix}_dir", train_dir)

    def init_logger(self, txt_logger_name, txt_logger_level):
        # Only main process creates the Tensorboard writer to avoid duplicate writes
        self.txt_logger = FusionLogger(
            logger_name=txt_logger_name,
            log_file=self.train_txt_logs_dir / "log.log",
            log_level=txt_logger_level,
        )
        if self.accelerator.is_main_process:
            self.backend_logger = SummaryWriter(self.train_backend_logs_dir)
        else:
            self.backend_logger = None

    def init_dataloader(self):
        # Expect user passed torch.utils.data.DataLoader objects already
        # Do not move data to device manually — accelerator.prepare will handle it
        self.train_dataloader = self._orig_train_dataloader
        self.val_dataloader = self._orig_val_dataloader

        # Prepare will wrap dataloaders, model, optimizer later together
        # but we can prepare dataloaders early too:
        # (we postpone prepare of model/optimizer until they exist)
        # We'll call accelerator.prepare(...) in init_model_and_optimizer_and_scheduler

    def init_model_and_optimizer_and_scheduler(self):
        # model: DON'T call .to(device)
        self.model = self._orig_model
        assert self._orig_optimizer is not None, "optimizer must be provided"
        self.optimizer = self._orig_optimizer
        self.scheduler = (
            self._orig_scheduler(self.optimizer) if self._orig_scheduler is not None else None
        )
        # PrePare
        objs_to_prepare = [self.model, self.optimizer, self.train_dataloader, self.val_dataloader]
        if self.scheduler is not None:
            objs_to_prepare.append(self.scheduler)
        prepared = self.accelerator.prepare(*objs_to_prepare)
        # prepared returns items in same order
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = prepared[:4]
        if self.scheduler is not None:
            self.scheduler = prepared[4]

        # post-prepare: if you need model.mode attribute
        # (unwrap_model in case model is DDP-wrapped internally)
        try:
            # if model has attribute 'mode' inside underlying module
            self.mode = getattr(self.accelerator.unwrap_model(self.model), "mode", None)
        except Exception:
            self.mode = None

    def init_metric(self):
        # metrics may be nn.Modules or callable; ensure they live on correct device
        # If metric is an nn.Module, move it to accelerator.device.
        processed_metrics = []
        for metric in self._orig_metric_list:
            if hasattr(metric, "to"):
                # move metric to correct device
                metric = metric.to(self.accelerator.device)
            processed_metrics.append(metric)
        self.metric_list = processed_metrics

    def init_tracker(self):
        key_list = ["loss"] + [getattr(metric, "__name__", metric.__class__.__name__) for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch["coarse_img_01"]
        coarse_img_02 = data_per_batch["coarse_img_02"]
        fine_img_01 = data_per_batch["fine_img_01"]
        fine_img_02 = data_per_batch["fine_img_02"]
        model_input_list = [coarse_img_01, coarse_img_02, fine_img_01, fine_img_02]
        return model_input_list

    def get_model_sampling_input(self, data_per_batch):
        coarse_img_01 = data_per_batch["coarse_img_01"]
        coarse_img_02 = data_per_batch["coarse_img_02"]
        fine_img_01 = data_per_batch["fine_img_01"]
        model_sampling_input_list = [coarse_img_01, coarse_img_02, fine_img_01]
        return model_sampling_input_list

    def get_loss_gt(self, data_per_batch):
        fine_img_02 = data_per_batch["fine_img_02"]
        # laplacian_pyramid is model method; ensure using unwrapped model for CPU ops maybe
        x1, x2, x3 = self.accelerator.unwrap_model(self.model).laplacian_pyramid(fine_img_02, levels=3)

        if self.mode == "x3":
            return x3
        elif self.mode == "x2+x3":
            x3_up = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
            return x3_up + x2
        elif self.mode == "x1+x2+x3":
            x3_up = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
            x2_up = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            return x3_up + x2_up + x1
        else:
            return fine_img_02

    def get_metrics_gt(self, data_per_batch):
        fine_img_02 = data_per_batch["fine_img_02"]
        x1, x2, x3 = self.accelerator.unwrap_model(self.model).laplacian_pyramid(fine_img_02, levels=3)

        if self.mode == "x3":
            return x3
        elif self.mode == "x2+x3":
            x3_up = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
            return x3_up + x2
        elif self.mode == "x1+x2+x3":
            x3_up = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=False)
            x2_up = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            return x3_up + x2_up + x1
        else:
            return fine_img_02

    def train(self):
        while self.current_epoch < self.max_epoch:
            self.train_epoch()
            if (self.current_epoch + 1) % self.save_interal == 0:
                self.save_checkpoint()
            if (self.current_epoch + 1) % self.val_interal == 0:
                self.val()
            self.current_epoch += 1

    def train_epoch(self):
        self.model.train()
        # set epoch for distributed sampler if exists
        try:
            sampler = getattr(self.train_dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.current_epoch)
        except Exception:
            pass

        self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            model_input_list = self.before_train_iter(data_per_batch)
            self.train_iter(iter_idx, model_input_list)
            self.current_train_step += 1
        self.ema.update()  # ema_pytorch update is called after optimizer.step() in train_iter
    
        if self.scheduler is not None:
            try: # per-epoch
                self.scheduler.step()
            except Exception:
                pass

        # logging (only main process)
        if self.accelerator.is_main_process and self.backend_logger is not None:
            msg = f'epoch: [{self.current_epoch + 1} / {self.max_epoch}]'
            current_lr = self.optimizer.param_groups[0]['lr']
            msg += f', lr: [{current_lr:.6f}]'
            self.backend_logger.add_scalar('train/lr', current_lr, self.current_epoch)

            for key, value in self.train_tracker.results.items():
                msg += f", {key}: {value:.4f}"
                self.backend_logger.add_scalar(f"train/{key}", value, self.current_epoch)
            self.txt_logger.info(msg)

    def before_train_iter(self, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        return model_input_list

    def train_iter(self, iter_idx, model_input_list):
        # Gradient accumulation handled by accelerator.accumulate
        # Use accumulate context to ensure optimizer.step executes only every N steps
        with self.accelerator.accumulate(self.model):
            # forward & loss
            with torch.no_grad():
                # If your model has internal randomness that must be enabled, remove torch.no_grad
                pass

            loss = self.model(*model_input_list)
            # use accelerator.backward for mixed precision and multi-gpu
            self.accelerator.backward(loss)
            # clip gradients on the unwrapped model parameters
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # optimizer step (will actually step only on accumulation boundary)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # logging
            if self.accelerator.is_main_process and self.backend_logger is not None:
                msg = f'epoch: [{self.current_epoch + 1} / {self.max_epoch}], iter: [{self.current_train_step + 1} / {self.max_train_iter}], loss: {loss.item():.4e}'
                self.train_tracker.update("loss", loss.item())
                self.backend_logger.add_scalar("train_running/loss", loss.item(), self.current_train_step)
                self.txt_logger.info(msg)

    def val(self):
        # use EMA model for evaluation (it is CPU/GPU tensor-copies inside EMA)
        self.accelerator.wait_for_everyone()  # sync before validation
        self.ema.ema_model.eval()
        self.val_tracker.reset()

        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            (
                model_sampling_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_val_iter(data_per_batch)

            self.val_iter(
                iter_idx,
                model_sampling_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )
            self.current_val_step += 1

        # aggregate and log on main process
        if self.accelerator.is_main_process and self.backend_logger is not None:
            msg = f"val epoch: {self.current_epoch}"
            for key, value in self.val_tracker.results.items():
                msg += f", {key}: {value:.4f}"
                self.backend_logger.add_scalar(f"val/{key}", value, self.current_epoch)
            self.txt_logger.info(msg)

    def before_val_iter(self, data_per_batch):
        model_sampling_input_list = self.get_model_sampling_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        # show_img_list keep cpu tensors for saving images later; move to cpu if needed
        show_img_list = [
            data_per_batch["coarse_img_01"].detach().cpu(),
            data_per_batch["coarse_img_02"].detach().cpu(),
            data_per_batch["fine_img_01"].detach().cpu(),
            data_per_batch["fine_img_02"].detach().cpu(),
        ]
        key = data_per_batch["key"]
        # normalize_scale and normalize_mode may be tensors on device — convert to cpu numpy scalars
        normalize_scale = data_per_batch["normalize_scale"][0].detach().cpu().numpy()
        normalize_mode = data_per_batch["normalize_mode"][0].detach().cpu().numpy()

        dataset_name = data_per_batch["dataset_name"][0]
        if isinstance(dataset_name, torch.Tensor):
            dataset_name = dataset_name.detach().cpu().numpy().tolist()

        return (
            model_sampling_input_list,
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
        model_sampling_input_list,
        loss_gt,
        metrics_gt,
        show_img_lists,
        keys,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        # compute outputs using EMA model
        with torch.no_grad():
            # ema.ema_model likely on same device as model parameters
            outputs = self.ema.ema_model.sample(*model_sampling_input_list)

            # compute pixel-wise loss on this device
            pixel_loss = F.mse_loss(outputs, loss_gt)
            self.val_tracker.update("loss", pixel_loss.item())

            # If multi-GPU: gather outputs and metrics_gt across processes before metric computation
            try:
                gathered_outputs = self.accelerator.gather(outputs)
                gathered_metrics_gt = self.accelerator.gather(metrics_gt)
            except Exception:
                # if gather not supported for some reason, fallback to local compute
                gathered_outputs = outputs
                gathered_metrics_gt = metrics_gt

            if self.accelerator.is_main_process:
                msg = f'val epoch: {self.current_epoch}, iter: [{iter_idx + 1}/{self.max_val_iter}]'
                msg += f', loss: {pixel_loss.item():.4e}'

            # compute metrics on gathered tensors (convert to proper scale)
            # metrics expect tensors on same device — ensure metrics are on accelerator.device
            for metric in self.metric_list:
                try:
                    metric_value = metric((gathered_outputs + 1.0) / 2.0, (gathered_metrics_gt + 1.0) / 2.0)
                except Exception:
                    metric_value = metric((gathered_outputs + 1.0).cpu() / 2.0, (gathered_metrics_gt + 1.0).cpu() / 2.0)

                metric_name = getattr(metric, "__name__", metric.__class__.__name__)
                self.val_tracker.update(metric_name, metric_value.item() if hasattr(metric_value, "item") else float(metric_value))

                if self.accelerator.is_main_process and self.backend_logger is not None:
                    msg = msg + f', {metric_name}: {metric_value.item():.4f}'
                    self.backend_logger.add_scalar(
                        f"val_runinng/{metric_name}", metric_value.item(), self.current_val_step
                    )

            # Log pixel loss to tensorboard (main process)
            if self.accelerator.is_main_process and self.backend_logger is not None:
                self.backend_logger.add_scalar("val_running/loss", pixel_loss.item(), iter_idx)
                self.txt_logger.info(msg)

            # Save images only on main process to avoid duplication / IO races
            if self.accelerator.is_main_process:
                
                batch_num = outputs.shape[0]
                for batch_idx in range(batch_num):
                    key = keys[batch_idx]
                    show_img_list = [s[batch_idx : batch_idx + 1] for s in show_img_lists]
                    output = outputs[batch_idx : batch_idx + 1].detach().cpu()
                    save_dir_prefix = f"{dataset_name}/{self.current_epoch}/save_img"
                    save_dir_path = self.train_imgs_dir / save_dir_prefix
                    save_name = key.split("-")[-1]
                    save_name = save_name[:8] + "_save_img_" + save_name[9:] + ".tif"
                    if "STIL" in save_dir_prefix:
                        save_name = key + ".tif"
                    self.img_save(output, save_dir_path, save_name, normalize_scale, normalize_mode)

                    show_dir_prefix = f"{dataset_name}/{self.current_epoch}/show_img/fine_img"
                    show_dir_path = self.train_imgs_dir / show_dir_prefix
                    show_name = key.split("-")[-1]
                    show_name = show_name[:8] + "_show_fine_img_" + show_name[9:] + ".png"
                    if "STIL" in save_dir_prefix:
                        show_name = key + ".png"
                    self.img_show(show_img_list, output, show_dir_path, show_name, normalize_mode)

    # -------------------------
    # I/O helpers
    # -------------------------
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
            img_interval * (w_num) + w * (w_num - 1) : img_interval * (w_num) + w * (w_num),
            :,
        ] = show_sub_img

        show_dir_path.mkdir(parents=True, exist_ok=True)
        io.imsave(show_dir_path / show_name, show_img)

    def save_checkpoint(self):
        # unwrap model to get real state_dict for saving
        model_to_save = self.accelerator.unwrap_model(self.model)
        data = {
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }
        save_path = self.train_checkpoints_dir / f"model_epoch_{self.current_epoch}.pth"
        # accelerator.save will only save on main process
        self.accelerator.save(data, str(save_path))
    
    def save_checkpoint(self):
        # data = {
        #     'model': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'ema': self.ema.state_dict(),
        # }
        # torch.save(
        #     data,
        #     self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
        # )
        data = {
            'epoch': self.current_epoch,  # 关键：保存当前 epoch
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'current_train_step': self.current_train_step, 
        }
        save_path = self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth'
        torch.save(data, save_path)
        self.txt_logger.info(f'Checkpoint saved to {save_path}')

    def resume_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.txt_logger.warning(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")
            return

        self.txt_logger.info(f"Loading checkpoint from {checkpoint_path} ...")
        
        # 加载数据
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 1. 恢复模型权重
        self.model.load_state_dict(checkpoint['model'])
        
        # 2. 恢复优化器状态 (包含动量等缓存信息)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 3. 恢复 EMA
        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
            
        # 4. 恢复 Scheduler (如果有)
        if self.scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        # 5. 恢复 Epoch 和 Step
        # 注意：保存的是已完成的 epoch，所以开始训练时要 +1
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        
        if 'current_train_step' in checkpoint:
            self.current_train_step = checkpoint['current_train_step']

        self.txt_logger.info(f"Resumed training from epoch {self.current_epoch}")
