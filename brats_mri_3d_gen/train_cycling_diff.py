from cycling_utils import Timer

timer = Timer()
timer.report('importing Timer')

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from cycling_utils import InterruptableDistributedSampler, Timer
# from loops import train_generator_one_epoch, evaluate_generator
from loops import train_diffusion_one_epoch, evaluate_diffusion
import utils

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training", add_help=add_help)

    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
    parser.add_argument("--gen-load-path", type=str, help="path of checkpoint", dest="gen_load_path") # for checkpointing
    parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
    # parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--data-path", default="/mnt/Datasets/Open-Datasets/MONAI", type=str, help="dataset path", dest="data_path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    # parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size", dest="batch_size")
    # parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    # parser.add_argument("--print-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
 
    return parser

timer.report('importing everything else')

def main(args, timer):

    # ## Distributed training prelims
    # if args.output_dir:
    #     utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args.distributed # don't support cycling when not distributed for simplicity

    device = torch.device(args.device)

    timer.report('preliminaries')

    # Maybe this will work?
    set_determinism(42)

    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    ## NEED TO COME BACK AND ALIGN WITH BRATS CONFIG
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            transforms.AddChanneld(keys=["image"]),
            transforms.EnsureTyped(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
            transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        ]
    )

    train_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="training", cache_rate=0.0,
        num_workers=4, download=False, seed=0, transform=train_transforms,
    )
    val_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="validation", cache_rate=0.0,
        num_workers=4, download=False, seed=0, transform=train_transforms,
    )

    timer.report('build datasets')

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    timer.report('build samplers')

    train_loader = DataLoader(train_ds, batch_size=3, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)
    check_data = first(train_loader) # Used later

    timer.report('build dataloaders')

    # Auto-encoder definition taken from the monai tutorial
    generator = AutoencoderKL(
        spatial_dims=3, # 3 dimensions
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64), #different channels
        latent_channels=3, # 3 channels
        num_res_blocks=1, #1 res block
        norm_num_groups=16, #half groups
        attention_levels=(False, False, True), # true instead of false
    )
    # saved_generator_checkpoint = torch.load("/output_brats_mri_2d_gen/exp_1645/checkpoint.isc", map_location="cpu")
    saved_generator_checkpoint = torch.load(args.gen_load_path, map_location="cpu")
    generator.load_state_dict(saved_generator_checkpoint["generator"])
    generator = generator.to(device)

    timer.report('generator to device')

    # # Discriminator definition
    # discriminator = PatchDiscriminator(
    #     spatial_dims=2, num_layers_d=3, num_channels=64, 
    #     in_channels=1, out_channels=1
    # )
    # discriminator = discriminator.to(device)

    # timer.report('discriminator to device')

    # Diffusion model (unet)
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=1,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
    )
    unet = unet.to(device)

    timer.report('unet to device')

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    # Autoencoder loss functions
    # adv_loss = PatchAdversarialLoss(criterion="least_squares")
    # perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    # perceptual_loss.to(device)

    # timer.report('loss functions')

    # Prepare for distributed training
    # generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    # discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    # generator_without_ddp = generator
    # discriminator_without_ddp = discriminator
    unet_without_ddp = unet
    if args.distributed:
        # generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu], find_unused_parameters=True)
        # discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu], find_unused_parameters=True)
        unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.gpu], find_unused_parameters=True)
        # generator_without_ddp = generator.module
        # discriminator_without_ddp = discriminator.module
        unet_without_ddp = unet.module

    timer.report('unet prepped for distribution')

    # Optimizers
    # optimizer_g = torch.optim.Adam(generator_without_ddp.parameters(), lr=1e-4)
    # optimizer_d = torch.optim.Adam(discriminator_without_ddp.parameters(), lr=5e-4)
    # optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)
    optimizer_u = torch.optim.Adam(unet_without_ddp.parameters(), lr=1e-4)

    timer.report('optimizers')

    # For mixed precision training
    # scaler_g = GradScaler()
    # scaler_d = GradScaler()
    scaler_u = GradScaler()

    timer.report('grad scalers')

    # Init tracking metrics
    train_images_seen = 0
    val_images_seen = 0
    epoch_loss = 0
    val_loss = 0

    # RETRIEVE GENERATOR CHECKPOINT FROM PREVIOUS JOB
    
    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        checkpoint = torch.load(args.resume, map_location="cpu")
    elif args.prev_resume and os.path.isfile(args.prev_resume):
        checkpoint = torch.load(args.prev_resume, map_location="cpu")
    if checkpoint is not None:
        args.start_epoch = checkpoint["epoch"]
        unet_without_ddp.load_state_dict(checkpoint["unet"])
        optimizer_u.load_state_dict(checkpoint["optimizer_u"])
        scaler_u.load_state_dict(checkpoint["scaler_u"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        train_images_seen = checkpoint["train_images_seen"]
        val_images_seen = checkpoint["val_images_seen"]
        # Metrics
        train_images_seen = checkpoint["train_images_seen"]
        val_images_seen = checkpoint["val_images_seen"]
        epoch_loss = checkpoint["epoch_loss"]
        val_loss = checkpoint["val_loss"]

    timer.report('checkpoint retrieval')

    # ## -- TRAINING THE AUTO-ENCODER - ##

    # n_gen_epochs = 100
    # gen_val_interval = 1

    # for epoch in range(args.start_epoch, n_gen_epochs):

    #     print('\n')
    #     print(f"EPOCH :: {epoch}")
    #     print('\n')

    #     with train_sampler.in_epoch(epoch):
    #         timer = Timer("Start training")
    #         generator, timer = train_generator_one_epoch(
    #             args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
    #             scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
    #             train_images_seen, val_images_seen, epoch_loss, gen_epoch_loss, disc_epoch_loss, val_loss
    #         )
    #         timer.report(f'training generator for epoch {epoch}')

    #         if epoch % gen_val_interval == 0: # Eval every epoch
    #             with val_sampler.in_epoch(epoch):
    #                 timer = Timer("Start evaluation")
    #                 timer = evaluate_generator(
    #                     args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
    #                     scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
    #                     train_images_seen, val_images_seen, epoch_loss, gen_epoch_loss, disc_epoch_loss, val_loss
    #                 )
    #                 timer.report(f'evaluating generator for epoch {epoch}')


    ## -- TRAINING THE DIFFUSION MODEL - ##

    n_diff_epochs = 200
    diff_val_interval = 1

    # Prepare LatentDiffusionInferer
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
    with torch.no_grad():
        with autocast(enabled=True):
            z = generator.encode_stage_2_inputs(check_data["image"].to(device))
    scale_factor = 1 / torch.std(z)
    # inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    timer.report('building inferer')

    for epoch in range(args.start_epoch, n_diff_epochs):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = Timer("Start training")
            unet, timer = train_diffusion_one_epoch(
                args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
                train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
            )
            timer.report(f'training unet for epoch {epoch}')

            if epoch % diff_val_interval == 0:
                with val_sampler.in_epoch(epoch):
                    timer = Timer("Start evaluation")
                    timer = evaluate_diffusion(
                        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
                        train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
                    )
                    timer.report(f'evaluating unet for epoch {epoch}')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
