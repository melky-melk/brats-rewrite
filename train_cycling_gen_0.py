from cycling_utils import TimestampedTimer

timer = TimestampedTimer()
timer.report('importing TimestampedTimer')

import os

import torch
import torch.distributed as dist
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader# , Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler# , autocast
from pathlib import Path

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator # , DiffusionModelUNet

# NOTE USE THESE METRICS TRACKER ONCE IT RUNS   
from cycling_utils import InterruptableDistributedSampler, MetricsTracker, TimestampedTimer
# from cycling_utils import InterruptableDistributedSampler, TimestampedTimer
# import loops_0
from loops_0 import train_generator_one_epoch, evaluate_generator
import utils

def get_args_parser(add_help=True):
    import argparse

    import argparse
    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training", add_help=add_help)
    parser.add_argument("--resume", type=str, help="path of checkpoint", required=True) # for checkpointing
    parser.add_argument("--prev-resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing
    parser.add_argument("--tboard-path", default=None, help="path for saving tensorboard logs", dest="tboard_path") # for checkpointing
    parser.add_argument("--data-path", default="/mnt/Datasets/Open-Datasets/MONAI", type=str, help="dataset path", dest="data_path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=5e-5, type=float, help="initial learning rate")
    parser.add_argument("--kl-weight",default=1e-6,type=float, help="kl loss weight for generator", dest="kl_weight")
    parser.add_argument("--perceptual-weight",default=1.0,type=float, help="perceptual loss weight for generator", dest="perceptual_weight")
    parser.add_argument("--adv-weight",default=0.5,type=float, help="adversarial loss weight for generator", dest="adv_weight")
    
    return parser

timer.report('importing everything else')


# the way we are training this is that every machine does some of the work and then consolidates with the main machine this is running on
# updating all of the values in the local vector, then we are going to take it gather them together to reduce and reset. 
# all reduce on a local object. everyone who has a self.local exchanges with every opther machine everyone elses machine. dumps into a binary. 
# so its going to sum everything up and add it together
# class MetricsTracker:

#     def __init__(self, metric_names):
#         self.map = {n:i for i,n in enumerate(metric_names)}
#         # every machine has its own self.local value, which is just an array with a bunch of 0's during the training process, these local variables are added to
#         self.local = torch.zeros(len(metric_names), dtype=torch.float16, requires_grad=False, device='cuda')
#         # self.agg meaning aggrigate which is a storage of all the values, a running total of the information collected thus far
#         # we only care about the main machines self.add
#         self.agg = torch.zeros(len(metric_names), dtype=torch.float16, requires_grad=False, device='cuda')
#         self.epoch_reports = []

#     def update(self, metrics: dict):
#         for n,v in metrics.items():
#             self.local[self.map[n]] += v
    
#     # this is where all of our values are being collated
#     def reduce_and_reset_local(self):
#         # Reduce over all nodes, add that to local store, and reset local
#         dist.all_reduce(self.local, op=dist.ReduceOp.SUM)
#         self.agg += self.local
#         self.local = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')
    
#     def end_epoch(self):
#         self.epoch_reports.append(self.agg)
#         self.local = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')
#         self.agg = torch.zeros(len(self.local), dtype=torch.float16, requires_grad=False, device='cuda')

#     def to(self, device):
#         self.local = self.local.to(device)
#         self.agg = self.agg.to(device)

def main(args, timer):

    utils.init_distributed_mode(args) # Sets args.distributed among other things
    assert args.distributed # don't support cycling when not distributed for simplicity

    device = torch.device(args.device)

    timer.report('preliminaries')

    # Maybe this will work?
    set_determinism(42)

    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
    # all of the transforms functions are from MONAI's bundle stuff, it essentially takes the data and converts it in cuch a way where it is usable to train by the model
    # taken from the 3dibm
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
    
    # saving the dataset as an object randomly samples one of those images to test the brain
    # the Task01 arg comes from what  the isc saved the dataset under
    train_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="training", cache_rate=0.0,
        num_workers=8, download=False, seed=0, transform=train_transforms,
    )

    # creates a validation one not stated in the tutorial, but adam did both of them in one so im taking from that
    val_ds = DecathlonDataset(
        root_dir=args.data_path, task="Task01_BrainTumour", section="validation", cache_rate=0.0,
        num_workers=8, download=False, seed=0, transform=train_transforms,
    )

    timer.report('build datasets')

    # data loader fpr both
    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    timer.report('build samplers')

    # Original trainer had batch size = 26. Using 9 nodes x batch size 3 = eff batch size = 27
    train_loader = DataLoader(train_ds, batch_size=3, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)
    # check_data = first(train_loader) # Used later

    timer.report('build dataloaders')

    # generator, autoencoder, and another model called discriminator, learning to tell when the generator produced an image as opposed to a real image. creating an adversary for the model to tell when it creates a real image. and keeps creating until its satisfyable enough

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
    generator = generator.to(device)

    timer.report('generator to device')

    # Discriminator definition taken from monai build
    discriminator = PatchDiscriminator(
        spatial_dims=3, 
        num_layers_d=3, 
        num_channels=32, 
        in_channels=1, 
        out_channels=1)
    discriminator = discriminator.to(device)

    timer.report('discriminator to device')

    # Autoencoder loss functions
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    # changes from monai
    perceptual_loss = PerceptualLoss(
        spatial_dims=3, 
        network_type="squeeze", 
        is_fake_3d=True, 
        fake_3d_ratio=0.2
    )
    perceptual_loss.to(device)

    timer.report('loss functions')

    # gen is generator and diff is difference
    # Prepare for distributed training
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)

    # copy of all of the weights and all of the models they have been running on each machine, those are tracked in the generator without ddp thing
    generator_without_ddp = generator
    discriminator_without_ddp = discriminator
    if args.distributed:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters necessary for monai training
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters necessary for monai training
        generator_without_ddp = generator.module
        discriminator_without_ddp = discriminator.module

    timer.report('models prepped for distribution')

    # Optimizers
    optimizer_g = torch.optim.Adam(generator_without_ddp.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator_without_ddp.parameters(), lr=5e-5)
    # optimizer_u = torch.optim.Adam(unet_without_ddp.parameters(), lr=1e-4)

    timer.report('optimizers')

    # For mixed precision training
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    # scaler_u = GradScaler()

    timer.report('grad scalers')

    # Init metric tracker
    train_metrics = MetricsTracker(["train_images_seen", "epoch_loss", "gen_epoch_loss", "disc_epoch_loss"])
    val_metrics = MetricsTracker(["val_images_seen", "val_loss"])
    metrics = {'train': train_metrics, 'val': val_metrics}

    # RETRIEVE CHECKPOINT
    Path(args.resume).parent.mkdir(parents=True, exist_ok=True)
    # checkpoint isnt saved here, it saves later
    checkpoint = None
    if args.resume and os.path.isfile(args.resume): # If we're resuming...
        checkpoint = torch.load(args.resume, map_location="cpu")
    elif args.prev_resume and os.path.isfile(args.prev_resume):
        checkpoint = torch.load(args.prev_resume, map_location="cpu")
    if checkpoint is not None:
        args.start_epoch = checkpoint["epoch"]
        generator_without_ddp.load_state_dict(checkpoint["generator"])
        discriminator_without_ddp.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scaler_g.load_state_dict(checkpoint["scaler_g"])
        scaler_d.load_state_dict(checkpoint["scaler_d"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        # Metrics
        metrics = checkpoint["metrics"]
        metrics["train"].to(device)
        metrics["val"].to(device)

    timer.report('checkpoint retrieval')

    ## -- TRAINING THE AUTO-ENCODER - ##

    n_gen_epochs = 200
    gen_val_interval = 1

    # actually starts the training for the epochs
    for epoch in range(args.start_epoch, n_gen_epochs):

        print('\n')
        print(f"EPOCH :: {epoch}")
        print('\n')

        with train_sampler.in_epoch(epoch):
            timer = TimestampedTimer("Start training")
            # takes from the loops.py it does the actual model training
            generator, timer, metrics = train_generator_one_epoch(
                args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
                scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer, metrics
            )
            timer.report(f'training generator for epoch {epoch}')

            if epoch % gen_val_interval == 0: # Eval every epoch
                with val_sampler.in_epoch(epoch):
                    timer = TimestampedTimer("Start evaluation")
                    timer, metrics = evaluate_generator(
                        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
                        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer, metrics
                    )
                    timer.report(f'evaluating generator for epoch {epoch}')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
