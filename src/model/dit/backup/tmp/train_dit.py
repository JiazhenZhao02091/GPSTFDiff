import random
import shutil
from einops import rearrange, repeat
from omegaconf import OmegaConf
from dataloader_utils import get_dataloader
import torch
from utils_kl import kl_get_generator
from utils_kl import kl_get_dynamics
from utils_common import get_version_number
from accelerate.utils import DistributedDataParallelKwargs

from utils.train_utils import (
    create_logger,
    get_latest_checkpoint,
    get_model,
    grad_clip,
    requires_grad,
    update_ema,
    wandb_runid_from_checkpoint,
)
import torch.distributed as dist
from copy import deepcopy
from time import time
import logging
import os
from tqdm import tqdm
import wandb

from utils_kl import kl_get_encode_decode_fn


from utils.train_utils_args import rankzero_logging_info
import hydra
from hydra.core.hydra_config import HydraConfig
import accelerate
from wandb_utils import array2grid_pixel, get_max_ckpt_from_dir
import socket
from utils_common import print_rank_0



#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    if args.debug:
        args.data.batch_size = 16
        args.ckpt_every = 10
        args.data.sample_fid_n = 1_00
        args.data.sample_fid_bs = 4
        args.data.sample_fid_every = 5
        args.data.sample_vis_every = 3
        args.data.sample_vis_n = 2
        args.shortcut.num_steps = 16
        print_rank_0("debug mode, using smaller batch size and sample size")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = accelerator.device
    accelerate.utils.set_seed(args.global_seed, device_specific=True)

    train_steps = 0

    #TODO 实验目录和checkpoint目录
    experiment_dir = HydraConfig.get().run.dir
    logging.info(f"Experiment directory created at {experiment_dir}")
    checkpoint_dir = (
        f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model, in_channels, input_size = get_model(args, device)

    if True:
        args.data.sample_fid_bs = min(30, args.data.batch_size // 2)
        # https://github.com/baofff/U-ViT/blob/ce551708dc9cde9818d2af7d84dfadfeb7bd9034/configs/imagenet512_uvit_large.py#L66C9-L66C27
        print_rank_0(
            f"forced sample_fid_bs to be equal to batch_size={args.data.sample_fid_bs}"
        )
        _fid_eval_batch_nums = args.data.sample_fid_n // (
            args.data.sample_fid_bs * accelerator.state.num_processes
        )
        assert _fid_eval_batch_nums > 0, f"{_fid_eval_batch_nums} <= 0"

    ema_model = deepcopy(model).to(device)

    # TODO 优化器
    if args.optim.name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd, fused=args.optim.fused
        )
    elif args.optim.name == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd, fused=args.optim.fused
        )
    else:
        raise ValueError(f"Optimizer {args.optim.name} not supported")

    update_ema(
        ema_model, model, decay=0
    ) 
    # 损失计算函数 和 采样函数
    training_losses_fn, sample_fn = kl_get_dynamics(args, device)

    _param_amount = sum(p.numel() for p in model.parameters())

    logger.info(f"#parameters: {_param_amount}")

    # TODO dataloader
    loader = get_dataloader(args)
    loader, opt, model, ema_model = accelerator.prepare(loader, opt, model, ema_model)

    if args.ckpt is not None:
        if os.path.isdir(args.ckpt):
            args.ckpt = get_max_ckpt_from_dir(args.ckpt)
            pass 
    if args.ckpt is not None:  # before accelerate.wrap()
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        opt.load_state_dict(state_dict["opt"])

        logging.info("overriding args with checkpoint args")
        logging.info(args)
        train_steps = state_dict["train_steps"]
        

        logging.info(f"Loaded checkpoint from {ckpt_path}, train_steps={train_steps}")
        requires_grad(ema_model, False)
        if rank == 0:
            shutil.copy(ckpt_path, checkpoint_dir)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_model.eval()  # EMA model should always be in eval mode

    log_steps = 0
    running_loss = 0
    start_time = time()

    sample_vis_n = args.data.sample_vis_n
    assert (
        sample_vis_n <= args.data.batch_size // 2
    ), f"{sample_vis_n} <= {args.data.batch_size}"

    zs_fixed = init_z(sample_vis_n, device, in_channels, input_size, args)
    print("zs_shape")
    
    if args.use_ema:
        print_rank_0("using ema model for sampling...")
        model_fn = accelerator.unwrap_model(ema_model).forward_without_cfg
    else:
        raise ValueError("use_ema must be True")
    # TODO vae

    train_dg, real_img_dg = kl_get_generator(
        loader, train_steps, accelerator, args, device
    )

    num_step_list= [1,2,3,4,6,8,12,15,64]
    num_step_bestfid_list = [666]*len(num_step_list)

    def sample_img(bs, args, zs=None, step_num=None):
        if zs is None:
            _zs = init_z(bs, device, in_channels, input_size, args)
        else:
            _zs = zs
        vis_config = dict()
        if  args.data.num_classes > 0 and "cfm" not in args.data.name:
            ys = torch.randint(0, args.data.num_classes - 1, (len(_zs),)).to(device)
            sample_model_kwargs = dict(y=ys)
        elif "cfm" in args.data.name:
            _, y = next(train_dg)
            y = y[:len(_zs)]
            assert len(y) == len(_zs)
            sample_model_kwargs = dict(y=y)
        else:
            sample_model_kwargs = dict()
        if step_num is  None:
            step_num = args.shortcut.num_steps
        sample_model_kwargs["step_num"] = step_num
        #print_rank_0("sample_model_kwargs: ", sample_model_kwargs)
        ##############
        try:
            samples = sample_fn(_zs, model_fn, **sample_model_kwargs)[-1]
        except Exception as e:
            logging.info("sample_fn error", exc_info=True)
            if accelerator.is_main_process:
                if "sampling_error" not in wandb_run.tags:
                    wandb_run.tags = wandb_run.tags + ("sampling_error",)
                    print_rank_0("sampling_error, wandb_run.tags:", wandb_run.tags)
            samples = torch.rand_like(_zs)

        accelerator.wait_for_everyone()

        samples = decode_fn(samples)

        out_sample_global = accelerator.gather(samples.contiguous())
        return out_sample_global, samples, vis_config

    
    from utils.my_metrics_offline import MyMetric_Offline as MyMetric
    my_metric = MyMetric(npz_real=args.data.npz_real)

    gt_img = next(real_img_dg)

    while train_steps < args.data.train_steps:
        if args.shortcut.use_repa:
            x, y, dino_feature = next(train_dg)

            zs = [dino_feature] # # [bs,256,768]
            model_kwargs = (
                dict(
                    y=y,
                )
                if y is not None
                else dict()
            )
            model_kwargs.update(
                    zs=zs)
        else:
            x, y = next(train_dg)
            model_kwargs = dict(y=y)

        model_kwargs["train_progress"] = train_steps / args.data.train_steps
        sc_kwargs = args.shortcut
        with accelerator.autocast():
            loss_dict = training_losses_fn(model=model, ema_model=ema_model, x1=x, sc_kwargs=sc_kwargs, **model_kwargs)
        loss = loss_dict["loss"].mean()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if isinstance(opt, list):#muon
            for optimizer in opt:
                optimizer.step()
                optimizer.zero_grad()
        else:
            opt.step()
            opt.zero_grad()
        update_ema(ema_model, model)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1

        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            if is_multiprocess:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / accelerator.state.num_processes
            if accelerator.is_main_process:
                logging.info(
                    f"(step={train_steps:07d}/{args.data.train_steps}), Train Loss: {avg_loss:.4f}, BS-1GPU: {len(x)} Train Steps/Sec: {steps_per_sec:.2f}, slurm_job_id: {slurm_job_id}, {experiment_dir}"
                )
                logging.info(wandb_sync_command)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                logging.info(latest_checkpoint)
                logging.info(wandb_project_url)
                logging.info(wandb_name)

            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "train_steps": train_steps,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

        if train_steps % args.data.sample_vis_every == 0 and train_steps > 0:

            zs_random = init_z(sample_vis_n, device, in_channels, input_size, args)
            out_sample_global_random, samples, vis_config = sample_img(
                bs=sample_vis_n, args=args, zs=zs_random
            )
            out_sample_global_fixed, samples, vis_config = sample_img(
                bs=sample_vis_n, args=args, zs=zs_fixed
            )
            
            if accelerator.is_main_process:
                wandb_dict.update(
                    {
                        "vis/sample_fixed": wandb.Image(
                            array2grid_pixel(out_sample_global_fixed[:16])
                        ),
                        "vis/sample_random": wandb.Image(
                            array2grid_pixel(out_sample_global_random[:16])
                        ),
                    }
                )

                wandb.log(
                    wandb_dict,
                    step=train_steps,
                )
                rankzero_logging_info(rank, "Generating samples done.")
            torch.cuda.empty_cache()


    model.eval()



if __name__ == "__main__":
    main()


