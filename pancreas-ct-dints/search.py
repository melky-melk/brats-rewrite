from utils import TimestampedTimer
timer = TimestampedTimer()
timer.report('importing Timer')

import argparse
import json
import logging
import monai
import numpy as np
import os
import random
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from datetime import datetime
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from monai.apps import download_and_extract
from monai.data import (
    ThreadDataLoader,
    decollate_batch,
)
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    AsDiscrete,
    CastToTyped,
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.data import Dataset, create_test_image_3d, list_data_collate, partition_dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism
from scipy import ndimage
from monai.networks.nets import TopologySearch, DiNTS
from monai.losses import DiceCELoss

from pathlib import Path
from monai.bundle import ConfigParser
from utils import InterruptableDistributedSampler, MetricsTracker, atomic_torch_save
import utils
from torch.cuda.amp import autocast, GradScaler
timer.report('imports')

def main(args, timer):

    timer.report("started main")

    utils.init_distributed_mode(args)
    assert args.distributed

    parser = ConfigParser()
    parser.read_config(args.conf)

    dist_arg = args.distributed

    args = {
        "start_epoch": 0,
        "resume": args.resume,
        "prev_resume": args.prev_resume,
        "tboard_path": args.tboard_path,
        "arch_ckpt_path": parser["arch_ckpt_path"],
	    "device" : "cuda",
        "amp": parser["amp"],
        "data_file_base_dir": parser["data_file_base_dir"],
        "data_list_file_path": parser["data_list_file_path"],
        "determ": parser["determ"],
        "learning_rate": parser["learning_rate"],
        "learning_rate_arch": parser["learning_rate_arch"],
        "learning_rate_milestones": np.array(parser["learning_rate_milestones"]),
        "num_images_per_batch": parser["num_images_per_batch"],
        "num_epochs": parser["num_epochs"],  # around 20k iterations
        "num_epochs_per_validation": parser["num_epochs_per_validation"],
        "num_epochs_warmup": parser["num_epochs_warmup"],
        "num_sw_batch_size": parser["num_sw_batch_size"],
        "output_classes": parser["output_classes"],
        "overlap_ratio": parser["overlap_ratio"],
        "patch_size_valid": parser["patch_size_valid"],
        "ram_cost_factor": parser["ram_cost_factor"],
	    "distributed": dist_arg,
        "world_size" : 0,
    }

    #assert args["distributed"] # don't support cycling when not distributed for simplicity
    device = torch.device(args["device"])

#A
    train_transforms = parser.get_parsed_content("transform_train")
    val_transforms = parser.get_parsed_content("transform_validation")

    # deterministic training
    if args["determ"]:
        set_determinism(seed=0)
#A
# specific data path
    with open(args["data_list_file_path"], "r") as f:
        json_data = json.load(f)
    list_train = json_data["training"]
    list_valid = json_data["validation"]

    # training data, orig bar args
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(args["data_file_base_dir"], list_train[_i]["image"])
        str_seg = os.path.join(args["data_file_base_dir"], list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    train_files = files
    random.shuffle(train_files)

    # validation data
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(args["data_file_base_dir"], list_valid[_i]["image"])
        str_seg = os.path.join(args["data_file_base_dir"], list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = files
    # val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
    #    dist.get_rank()
    # ]
    # print("val_files:", len(val_files))
    timer.report("load data")

# A, sets default values for data loaders, orig specified each
# different dataset loader to enable checkpointing progress
# brought forward compared to the orig
    n_workers = 1
    cache_rate = 0.0

    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=n_workers)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=n_workers)

    train_sampler = InterruptableDistributedSampler(train_ds)
    val_sampler = InterruptableDistributedSampler(val_ds)

    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1)

# setup model, brought forward
# network architecture

  #orig
    dints_space = monai.networks.nets.TopologySearch(channel_mul=0.5, num_blocks=12,num_depths=4,use_downsample=True,device=device)
    model = DiNTS(dints_space, in_channels=1, num_classes=3, use_downsample=True)
    loss_func = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, squared_pred=True, batch=True, smooth_nr=1e-05, smooth_dr=1e-05)

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #orig
    # defines functions for eval_search, ensures input is a tensor and converts to suitable
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=args["output_classes"])])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=args["output_classes"])])

    # MC/A, from fashion_mnist
    # generates gpu ID
    device_id = dist.get_rank() % torch.cuda.device_count()
    
    
    #A
    model_without_ddp = model
    if args["distributed"]:
        model = DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimisers, orig bar args format
    # orig had .weight_parameters, which is not a thing
    # added world size setter
    args["world_size"] = utils.get_world_size()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args["learning_rate"] * args["world_size"], momentum=0.9, weight_decay=0.00004
    )
    arch_optimizer_a = torch.optim.Adam(
        [dints_space.log_alpha_a], lr=args["learning_rate"] * args["world_size"], betas=(0.5, 0.999), weight_decay=0.0
    )
    arch_optimizer_c = torch.optim.Adam(
        [dints_space.log_alpha_c], lr=args["learning_rate"] * args["world_size"], betas=(0.5, 0.999), weight_decay=0.0
    )

    #amp
    # Automatic mixed precision - use 32b float for accurate stuff, then for the dirty quick parts use 16bit float - CGPT
    if args["amp"]:

        scaler = GradScaler()
        if dist.get_rank() == 0 or torch.cuda.device_count() == 1: # A only adds condition 2
            print("[info] amp enabled")

    train_metrics = MetricsTracker()
    
    val_metric = torch.zeros((args["output_classes"] - 1) * 2, dtype=torch.float, device=device)

    #for later
    args["device"] = device

# --- GET CHECKPOINT
# Don't get earlier bc we are just fetching the state of the loaders etc, not the entire data cache
    Path(args["resume"]).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    if args["resume"] and os.path.isfile(args["resume"]): # If we're resuming...
        checkpoint = torch.load(args["resume"], map_location="cpu")
    elif args["prev_resume"] and os.path.isfile(args["prev_resume"]):
        checkpoint = torch.load(args["prev_resume"], map_location="cpu")
    if checkpoint is not None:
        args["start_epoch"] = checkpoint["epoch"]
        model_without_ddp.load_state_dict(checkpoint["model"])
        dints_space.load_state_dict(checkpoint["dints"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        arch_optimizer_a.load_state_dict(checkpoint["arch_optimizer_a"])
        arch_optimizer_c.load_state_dict(checkpoint["arch_optimizer_c"])
        train_sampler.load_state_dict(checkpoint["train_sampler"])
        val_sampler.load_state_dict(checkpoint["val_sampler"])
        scaler.load_state_dict(checkpoint["scaler"])
        train_metrics = checkpoint["train_metrics"]
        val_metric = checkpoint["val_metric"]
        val_metric.to(device)

        timer.report("loading checkpoint")

# bits
    val_interval = args["num_epochs_per_validation"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    idx_iter = 0
    metric_values = list()

# --- Epoch loop
    
    start_time = time.time()
    for epoch in range(args["start_epoch"], args["num_epochs"]):

        timer.report(f"started epoch {epoch}")

        with train_sampler.in_epoch(epoch):
            timer = TimestampedTimer("Start training")

            model, dints_space, timer, train_metrics = search_one_epoch(
                model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c,
                train_sampler, val_sampler, scaler, train_metrics, val_metric,
                epoch, train_loader, loss_func, args, timer
            )
            timer.report(f'searching space for epoch {epoch}')

            if (epoch + 1) % val_interval == 0 or (epoch + 1) == args["num_epochs"]:

                with val_sampler.in_epoch(epoch):
                    timer = TimestampedTimer("Start evaluation")

                    timer = eval_search(
                        model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c,
                        train_sampler, val_sampler, scaler, train_metrics, val_metric,
                        epoch, val_loader, post_pred, post_label, args, timer
                    )
                    timer.report(f'full cycle for epoch {epoch}')
# --- END

        # removed some dict_file saving for metrics
        # timing stuff

# --- END MAIN ----


def search_one_epoch(
    model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c,
    train_sampler, val_sampler, scaler, train_metrics, val_metric,
    epoch, train_loader, loss_func, args, timer):

    decay = 0.5 ** np.sum(
        [(epoch - args["num_epochs_warmup"]) / (args["num_epochs"] - args["num_epochs_warmup"]) > args["learning_rate_milestones"]]
    )
    lr = args["learning_rate"] * decay * args["world_size"]#A add world size
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    #Removed metric printer
    model.train()

    #SC stepping/checkpointing stuff
    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    #removed epoch/torch loss. loss_torch_arch = torch.zeroes(...)


# --- Batch loop

    for batch_data in train_loader:

        print(f'Started batch {train_step}')

# REMOVE FOR REAL VER
        # for speeding up eval_search error testing
        #if train_step > 2:
         #   print(f'!!!!!         Skipping from batch {train_step} to get to eval_search faster')
          #  break

        inputs, labels = batch_data["image"].to(args["device"]), batch_data["label"].to(args["device"])
        # Adam: added, will this work? Matan: yes
        inputs_search, labels_search = inputs.detach().clone(), labels.detach().clone()

        # UPDATE MODEL
        # Orig had conditional on world size, using worker modules/not
        for p in model.module.weight_parameters():
            p.requires_grad=True
        dints_space.log_alpha_a.requires_grad = False
        dints_space.log_alpha_c.requires_grad = False

        optimizer.zero_grad()

        #Orig
        if args["amp"]:
            with autocast():
                outputs = model(inputs)
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if args["output_classes"] == 2:
                loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
            else:
                loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        #removed some loss metric vars and reporting

        train_metrics.update({"model_loss": loss.item(), "inputs_seen": len(inputs)})

        #flipped condition
        if epoch >= args["num_epochs_warmup"]:

            #removed dataloader.next stuff, labeling for it

            #update model, same differences as before
            for p in model.module.weight_parameters():
                p.requires_grad=False
            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            # Orig linear increase topology and RAM loss, unused though
            '''entropy_alpha_c = torch.tensor(0.0).to(device)
            entropy_alpha_a = torch.tensor(0.0).to(device)
            ram_cost_full = torch.tensor(0.0).to(device)
            ram_cost_usage = torch.tensor(0.0).to(device)
            ram_cost_loss = torch.tensor(0.0).to(device)
            topology_loss = torch.tensor(0.0).to(device)'''

            #Orig
            probs_a, arch_code_prob_a = dints_space.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()

            #Orig had sm and lsm in the declaration
            # softmax is for probability distribution, checks diff between predicted classes and expected
            # ask the robot for deets
            sm = F.softmax(dints_space.log_alpha_c, dim=-1)
            lsm = F.log_softmax(dints_space.log_alpha_c, dim=-1)
            entropy_alpha_c = -(sm * lsm).mean()
            topology_loss = dints_space.get_topology_entropy(probs_a)

            # Orig
            ram_cost_full = dints_space.get_ram_cost_usage(inputs.shape, full=True)
            ram_cost_usage = dints_space.get_ram_cost_usage(inputs.shape)
            ram_cost_loss = torch.abs(args["factor_ram_cost"] - ram_cost_usage / ram_cost_full)
            arch_optimizer_a.zero_grad()
            arch_optimizer_c.zero_grad()

            combination_weights = (epoch - args["num_epochs_warmup"]) / (args["num_epochs"] - args["num_epochs_warmup"])

            # Orig
            if args["amp"]:
                with autocast():
                    outputs_search = model(inputs_search)
                    if args["output_classes"] == 2:
                        loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                    else:
                        loss = loss_func(outputs_search, labels_search)

                loss += combination_weights * (
                    (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                scaler.scale(loss).backward()
                scaler.step(arch_optimizer_a)
                scaler.step(arch_optimizer_c)
                scaler.update()
            else:
                outputs_search = model(inputs_search)
                if args["output_classes"] == 2:
                    loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                else:
                    loss = loss_func(outputs_search, labels_search)

                loss += 1.0 * (
                    combination_weights * (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                loss.backward()
                arch_optimizer_a.step()
                arch_optimizer_c.step()
            
            train_metrics.update({"space_loss": loss.item()})
            print('loss.item is ' + loss.item +'.')
            # removed torch_arch metrics, like before
            # removed writing of metrics

# --- End batch loop

        # batch reporting
        train_metrics.reduce()
        batch_model_loss = train_metrics.local["model_loss"] / train_metrics.local["inputs_seen"]
        if "space_loss" in train_metrics.local:
            batch_space_loss = train_metrics.local["space_loss"] / train_metrics.local["inputs_seen"]
        else:
            batch_space_loss = "NONE"
        
        if batch_space_loss != "NONE":
            print(f"EPOCH [{epoch}], BATCH [{train_step}], MODEL LOSS [{batch_model_loss:.3f}], SPACE LOSS: [{batch_space_loss:.3f}]")
        
        train_metrics.reset_local()

        print(f'batch model loss {batch_model_loss}')

# --- Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(inputs))
        train_step = train_sampler.progress // train_loader.batch_size

        if train_step == total_steps:
            train_metrics.end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch

            writer = SummaryWriter(log_dir=args["tboard_path"])
            writer.add_scalar("Train/model_loss", batch_model_loss, train_step + epoch * total_steps)
            if batch_space_loss != "NONE":
                writer.add_scalar("Train/space_loss", batch_space_loss, train_step + epoch * total_steps)
            writer.flush()
            writer.close()

            checkpoint = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "dints": dints_space.state_dict(),
                "optimizer": optimizer.state_dict(),
                "arch_optimizer_a": arch_optimizer_a.state_dict(),
                "arch_optimizer_c": arch_optimizer_c.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_metrics": train_metrics,
                "val_metric": val_metric
            }
            timer = atomic_torch_save(checkpoint, args["resume"], timer)

    return model, dints_space, timer, train_metrics


def eval_search(
    model, optimizer, dints_space, arch_optimizer_a, arch_optimizer_c,
    train_sampler, val_sampler, scaler, train_metrics, val_metric,
    epoch, val_loader, post_pred, post_label, args, timer):

    timer.report(f'starting eval search {epoch}')

    #pulled this out from the branches
    torch.cuda.empty_cache()
    model.eval()

    #removed loss_torch_arch stuff    

    with torch.no_grad():

        #val step is currently defaulting to 0just s
        val_step = val_sampler.progress // val_loader.batch_size
        total_steps = int(len(val_sampler) / val_loader.batch_size)
        print(f'\neval_search: evaluating / resuming epoch {epoch} from eval step {val_step}')
        print(f'eval_search: sampler {val_sampler.progress}, step {val_step}')
        #removed some helper vars
        val_images = None
        val_labels = None
        val_outputs = None
        best_metric = -1
        best_metric_epoch = -1
        idx_iter = 0

        best_metric_iterations = -1

        device  = args["device"]

        for val_data in val_loader:
            print(f'eval_search: loaded val data {val_data}')
            # Orig
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            roi_size = args["patch_size_valid"]
            sw_batch_size = args["num_sw_batch_size"]

            print('eval_search: fixed data')

            # Orig
            if args["amp"]:
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(
                        val_images,
                        roi_size,
                        sw_batch_size,
                        lambda x: model(x),
                        mode="gaussian",
                        overlap=args["overlap_ratio"],
                    )
            else:
                pred = sliding_window_inference(
                    val_images,
                    roi_size,
                    sw_batch_size,
                    lambda x: model(x),
                    mode="gaussian",
                    overlap=args["overlap_ratio"],
                )
            #
            print('eval_search: got predictions')
            val_outputs = pred

            # Orig
            val_outputs = post_pred(val_outputs[0, ...])
            val_outputs = val_outputs[None, ...]
            val_labels = post_label(val_labels[0, ...])
            val_labels = val_labels[None, ...]

            value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=False)

            print("eval_search - computed dice.")
            # Removed metric vals

            # Orig
            for _c in range(args["output_classes"] - 1):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, 0]).float()

                # me, get all tensors onto the same device. jank? 
                if val0.device != torch.device('cpu'):
                    val1 = val1.to(val0.device)
                    if isinstance(val_metric, torch.Tensor):
                        val_metric= val_metric.to(val0.device)
                    print(f'collected to val0, {val0.device}')

                elif val1.device != torch.device('cpu'):
                    val0 = val0.to(val1.device)
                    if isinstance(val_metric, torch.Tensor):
                        val_metric= val_metric.to(val1.device)
                    print(f'collected to val1, {val1.device}')
                
    
                # me
                # dense tensors are filled out, instead of just index of filled values
                # .all_reduce synchronises the tensor across devices 
                val1 = val1.to_dense()
                val0 = val0.to_dense()

                # try remove
                #dist.all_reduce(val0, op=dist.ReduceOp.SUM)
                #dist.all_reduce(val1, op=dist.ReduceOp.SUM)
                #dist.all_reduce(val_metric, op=dist.ReduceOp.SUM)

                #Testing for dense, correct device
                if isinstance(val0, torch.Tensor) and isinstance(val1, torch.Tensor) and isinstance(val_metric, torch.Tensor):
                    print('eval_search: no tensors sparse.')
                print(f'val0 {val0.device}, val1 {val1.device}, val_metric {type(val_metric)}')

                #print(f'eval_search - collected tensors to {val_metric.device}')

                val_metric[2 * _c] += val0 * val1
                val_metric[2 * _c + 1] += val1

            # Checkpoint
            print(f"eval_search: Saving checkpoint at epoch {epoch} eval batch {val_step}")
            val_sampler.advance(len(val_images))
            val_step = val_sampler.progress // val_loader.batch_size

            if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch

                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "dints": dints_space.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "arch_optimizer_a": arch_optimizer_a.state_dict(),
                    "arch_optimizer_c": arch_optimizer_c.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "val_sampler": val_sampler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metric": val_metric
                    # missing anything?
                }
                timer = atomic_torch_save(checkpoint, args["resume"], timer)
            print('eval_search: saved checkpoint.')
            # synchronizes all processes and reduce results

            # Added condition
            if torch.cuda.device_count() > 1:
                dist.barrier()
                dist.all_reduce(val_metric, op=torch.distributed.ReduceOp.SUM)

            val_metric = val_metric.tolist()

            # condition in original checked for rank
            if utils.is_main_process():
                for _c in range(args["output_classes"] - 1):
                    print("evaluation metric - class {0:d}:".format(_c + 1), val_metric[2 * _c] / val_metric[2 * _c + 1])
                avg_metric = 0
                for _c in range(args["output_classes"] - 1):
                    avg_metric += val_metric[2 * _c] / val_metric[2 * _c + 1]
                avg_metric = avg_metric / float(args["output_classes"] - 1)
                print("avg_metric", avg_metric)

                if avg_metric > best_metric:
                    best_metric = avg_metric
                    best_metric_epoch = epoch + 1
                    best_metric_iterations = idx_iter

                print('eval_search: started saving metric to search_code.pt')
                node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d = dints_space.decode()
                torch.save(
                    {
                        "node_a": node_a_d,
                        "arch_code_a": arch_code_a_d,
                        "arch_code_a_max": arch_code_a_max_d,
                        "arch_code_c": arch_code_c_d,
                        "iter_num": idx_iter,
                        "epochs": epoch + 1,
                        "best_dsc": best_metric,
                        "best_path": best_metric_iterations,
                    },

                    #Edited this
                    os.path.join(args["arch_ckpt_path"], "search_code.pt"),
                )
    print(f'Completed eval_search for epoch {epoch}!')

    return timer

def get_args_parser():
    parser = argparse.ArgumentParser(description="Pancreas CT DiNTS Model Training")

    parser.add_argument(
        "--factor_ram_cost",
        default=0.0,
        type=float,
        help="factor to determine RAM cost in the searched architecture",
	required=False
    )
    parser.add_argument(
        "--fold",
        action="store",
        required=False,
        help="fold index in N-fold cross-validation",
    )
    parser.add_argument(
        "--json",
        action="store",
        required=False,
        help="full path of .json file",
    )
    parser.add_argument(
        "--json_key",
        action="store",
        required=False,
        help="selected key in .json data list",
    )
    parser.add_argument(
        "--local_rank",
        help="local process rank",
	required=False
    )
    parser.add_argument(
        "--num_folds",
        action="store",
        required=False,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--output_root",
        action="store",
        required=False,
        help="output root",
    )
    parser.add_argument(
        "--root",
        action="store",
        required=False,
        help="data root",
    )
# --- Added
    #parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--conf", type = str, help = "path of config", default = "configs/search.yaml", required=False)
    parser.add_argument("--resume", type =str, help="path to checkpoint", required=False)
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint full path", required=False)
    parser.add_argument("--tboard-path", default=None, help="path for saving tensorboard logs", dest="tboard_path", required=False) # for checkpointing
    parser.add_argument("--dist_url", type=str, default="env://", required=False)
    parser.add_argument("--prev_resume", default=None, help="path of previous job checkpoint for strong fail resume", dest="prev_resume") # for checkpointing    return parser
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
