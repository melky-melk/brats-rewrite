from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
import utils
from cycling_utils import atomic_torch_save
from torch.utils.tensorboard import SummaryWriter
from generative.losses.adversarial_loss import PatchAdversarialLoss
from torchvision.utils import make_grid
# tb_path = "/mnt/Client/StrongUniversity/USYD-04/usyd04_adam/output_brats_mri_2d_gen/tb"
tb_path = "/mnt/Client/Ctan682hia6krkvbghfcwq2mtmmnzsxa/ctactafmnaa6edh5dendtkiwzdi5fosy"

def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]

def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, 
        dim=list(range(1, len(z_sigma.shape)))
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

# def generator_loss(gen_images, real_images, z_mu, z_sigma, perceptual_loss, kl_weight, perceptual_weight):
#     # Image intrinsic qualities
#     # recons_loss = intensity_loss(gen_images, real_images)
#     recons_loss = F.l1_loss(gen_images.float(), real_images.float())
#     kl_loss = compute_kl_loss(z_mu, z_sigma)
#     p_loss = perceptual_loss(gen_images.float(), real_images.float())
#     loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
#     return loss_g

    # # Image intrinsic qualities
    # recons_loss = intensity_loss(gen_images, real_images)
    # kl_loss = compute_kl_loss(z_mu, z_sigma)
    # p_loss = perceptual_loss(gen_images.float(), real_images.float())
    # loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
    # # Discrimnator-based loss
    # logits_fake = disc_net(gen_images)[-1]
    # generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    # loss_g = loss_g + (adv_weight * generator_loss)

# def discriminator_loss(gen_images, real_images, discriminator, adv_weight):
#     logits_fake = discriminator(gen_images.contiguous().detach())[-1]
#     loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
#     logits_real = discriminator(real_images.contiguous().detach())[-1]
#     loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
#     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

#     loss_d = adv_weight * discriminator_loss
#     return loss_d

## -- AUTO-ENCODER - ##
# this training sequence was taken offf of monai's tutorial code
def train_generator_one_epoch(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
        metrics
    ):

    # Maybe pull these out into args later
    kl_weight = 1e-6
    generator_warm_up_n_epochs = 3
    perceptual_weight = 0.001
    adv_weight = 0.01

    generator.train()
    discriminator.train()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    # these are going to be very different between the models
    for step, batch in enumerate(train_loader):
        
        images = batch["image"].to(device)
        timer.report(f'train batch {train_step} to device')

        #####################################TRAIN GENERATOR###################################

        optimizer_g.zero_grad(set_to_none=True)

        reconstruction, z_mu, z_sigma = generator(images)
        timer.report(f'train batch {train_step} generator forward')

        # recons_loss = F.l1_loss(reconstruction.float(), images.float())
        # loss_g = generator_loss(reconstruction, images, z_mu, z_sigma, perceptual_loss, kl_weight, perceptual_weight)

        kl_loss = KL_loss(z_mu, z_sigma)

        recons_loss = F.l1_loss(reconstruction.float(), images.float())
        p_loss = perceptual_loss(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        # loss_g = generator_loss(reconstruction, images, z_mu, z_sigma, perceptual_loss, args.kl_weight, args.perceptual_weight)

        # discriminator based loss only occurs with this condition? idk it was in the tutorial
        if epoch > generator_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss

        timer.report(f'train batch {train_step} generator loss: {loss_g.item():.3f}')

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        # loss_g.backward()
        # optimizer_g.step()
        timer.report(f'train batch {train_step} generator backward')

        ##########################TRAIN DISCRIMINATOR######################
        # if epoch > generator_warm_up_n_epochs:  # Train generator for n epochs before starting discriminator training
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss
        timer.report(f'train batch {train_step} discriminator loss {loss_d.item():.3f}')
        # NOTE SCALAR STUFF IS COMMENTED OUT ADD BACK IN LATER WHEN ITS WORKING
        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
        # loss_d.backward()
        # optimizer_d.step()
        timer.report(f'train batch {train_step} discriminator backward')

        # NOTE TAKEN FROM THE TUTORIAL BUT DOESNT APPLY HERE 
        # epoch_loss += recons_loss.item()
        # if epoch > generator_warm_up_n_epochs
        #     gen_epoch_loss += generator_loss.item()
        #     disc_epoch_loss += discriminator_loss.item()
        
        # Reduce metrics accross nodes
        metrics["train"].update({"train_images_seen":len(images), "loss_g":loss_g.item(), "loss_d": loss_d.item()})
        metrics["train"].reduce()

        timer.report(f'train batch {train_step} metrics update')


        gen_loss = metrics["train"].local["loss_g"] / metrics["train"].local["train_images_seen"]
        disc_loss = metrics["train"].local["loss_d"] / metrics["train"].local["train_images_seen"]
        print("Epoch [{}] Step [{}/{}], gen_loss: {:.3f}, disc_loss: {:.3f}".format(epoch, train_step, total_steps, gen_loss, disc_loss))
        # recons_loss = metrics["train"].local["loss"] / metrics["train"].local["images_seen"] 
        # gen_loss = metrics["train"].agg[metrics["train"].map["gen_epoch_loss"]] / metrics["train"].agg[metrics["train"].map["train_images_seen"]]
        # disc_loss = metrics["train"].agg[metrics["train"].map["disc_epoch_loss"]] / metrics["train"].agg[metrics["train"].map["train_images_seen"]]
        metrics["train"].reset_local()

        timer.report(f'train batch {train_step} metrics update')
        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size
        
        if train_step == total_steps:
            metrics["train"].end_epoch()

        # if our crrent one is the main process, then we checkpoint it rather than checkpointing every single different machine
        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            writer = SummaryWriter(log_dir=tb_path)
            writer.add_scalar("recons_loss", recons_loss, step)
            writer.add_scalar("gen_loss", recons_loss, step)
            writer.add_scalar("disc_loss", recons_loss, step)
            writer.flush()
            writer.close()
            checkpoint = {
                # Universals
                "args": args,
                "epoch": epoch,
                # State variables
                "generator": generator.module.state_dict(),
                "discriminator": discriminator.module.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                # Metrics
                "metrics": metrics,
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    gen_loss = metrics["train"].epoch_reports[-1]["loss_g"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    disc_loss = metrics["train"].epoch_reports[-1]["loss_d"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    print("Epoch [{}] :: gen_loss: {:,.3f}, disc_loss: {:,.3f}".format(epoch, gen_loss, disc_loss))
    return generator, timer, metrics

# is just checkpoint?? taking the L1 loss comparison between the input and putput images
# evaluate distinct from training, training on dataset, then is our model generalising to data it hasnt seen. is it overfitting
# the evaluate is that you give it a different set of data to make predictions on havent updated the weights and stuff
# from the functional library. also logging the reconstructions from tensor board, giving qualitative picture
# tensorboard runs locally.
# this is where the reconstruction is evaluated after the epoch, the generator loss and discriminator is done in the above function
# def evaluate_generator(
#         args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
#         scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
#         metrics
#     ):

#     generator.eval()

#     val_step = val_sampler.progress // val_loader.batch_size
#     total_steps = int(len(val_sampler) / val_loader.batch_size)
#     print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

#     with torch.no_grad():
#         for batch in val_loader:

#             images = batch["image"].to(device)
#             timer.report(f'eval batch {val_step} to device')

#             with autocast(enabled=True):

#                 reconstruction, _, _ = generator(images)
#                 timer.report(f'eval batch {val_step} forward')
#                 recons_loss = F.l1_loss(images.float(), reconstruction.float())
#                 timer.report(f'eval batch {val_step} recons_loss')

#             metrics["val"].update({"val_images_seen": len(images), "val_loss": recons_loss.item()})
#             metrics["val"].reduce()
#             metrics["val"].reset_local()

#             timer.report(f'eval batch {val_step} metrics update')

#             ## Checkpointing
#             print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
#             val_sampler.advance(len(images))
#             val_step = val_sampler.progress // val_loader.batch_size

#             if val_step == total_steps:
#                  metrics["val"].end_epoch()

#             if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch
#                 print(f"Saving checkpoint at epoch {epoch} train batch {val_step}")
#                 checkpoint = {
#                     # Universals
#                     "args": args,
#                     "epoch": epoch,
#                     # State variables
#                     "generator": generator.module.state_dict(),
#                     "discriminator": discriminator.module.state_dict(),
#                     "optimizer_g": optimizer_g.state_dict(),
#                     "optimizer_d": optimizer_d.state_dict(),
#                     "scaler_g": scaler_g.state_dict(),
#                     "scaler_d": scaler_d.state_dict(),
#                     "train_sampler": train_sampler.state_dict(),
#                     "val_sampler": val_sampler.state_dict(),
#                     # Metrics
#                     "metrics": metrics,
#                 }
#                 timer = atomic_torch_save(checkpoint, args.resume, timer)

#     # val_loss = metrics["val"].agg[metrics["val"].map["val_loss"]] / metrics["val"].agg[metrics["val"].map["val_images_seen"]]
#     val_loss = metrics["val"].epoch_reports[-1]["loss"] / metrics["val"].epoch_reports[-1]["images_seen"]
#     if utils.is_main_process():
#         writer = SummaryWriter(log_dir=tb_path)
#         writer.add_scalar("val", val_loss, epoch)
#         writer.flush()
#         writer.close()
#     print(f"Epoch {epoch} val loss: {val_loss:.4f}")

#     return timer, metrics


def evaluate_generator(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, val_loader, device, timer, metrics
    ):

    generator.eval()

    val_step = val_sampler.progress // val_loader.batch_size
    total_steps = int(len(val_sampler) / val_loader.batch_size)
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

    with torch.no_grad():
        for batch in val_loader:

            images = batch["image"].to(device)
            timer.report(f'eval batch {val_step} to device')

            with autocast(enabled=True):

                reconstruction, _, _ = generator(images)
                timer.report(f'eval batch {val_step} forward')
                recons_loss = F.l1_loss(images.float(), reconstruction.float())
                timer.report(f'eval batch {val_step} recons_loss')

            metrics["val"].update({"val_images_seen": len(images), "val_loss": recons_loss.item()})
            metrics["val"].reduce()
            metrics["val"].reset_local()

            timer.report(f'eval batch {val_step} metrics update')

            ## Checkpointing
            print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
            val_sampler.advance(len(images))
            val_step = val_sampler.progress // val_loader.batch_size

            if val_step == total_steps:

                val_loss = metrics["val"].agg["val_loss"] / metrics["val"].agg["val_images_seen"]
                if utils.is_main_process():

                    writer = SummaryWriter(log_dir=args.tboard_path)
                    writer.add_scalar("Val/loss", val_loss, epoch)

                    images_list = torch.zeros((11*6, *images.shape[1:]), device=device, dtype=images.dtype)
                    reconstruction_list = torch.zeros((11*6, *reconstruction.shape[1:]), device=device, dtype=reconstruction.dtype)
                    dist.all_gather_into_tensor(images_list, images.clone())
                    dist.all_gather_into_tensor(reconstruction_list, reconstruction)
                    plottable = torch.cat((images_list[0:5],reconstruction_list[0:5]))
                    plottable = (plottable * 255).to(torch.uint8)


                    plottable = torch.cat((images, reconstruction))
                    grid = make_grid(plottable, nrow=2)
                    writer.add_image('Val/images', grid, epoch)

                    writer.flush()
                    writer.close()

                print(f"Epoch {epoch} val loss: {val_loss:.4f}")
                metrics["val"].end_epoch()

            if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch
                checkpoint = {
                    # Universals
                    "args": args,
                    "epoch": epoch,
                    # State variables
                    "generator": generator.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "val_sampler": val_sampler.state_dict(),
                    # Metrics
                    "metrics": metrics,
                }
                timer = atomic_torch_save(checkpoint, args.resume, timer)

    return timer, metrics

## -- DIFFUSION MODEL - ##

def train_diffusion_one_epoch(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
    ):

    unet.train() #shouldnt this be within the epocghs not on tyhe outside??
    generator.eval() 

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for step, batch in enumerate(train_loader):

        images = batch["image"].to(device)
        timer.report(f'train batch {train_step} to device')

        optimizer_u.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            z_mu, z_sigma = generator.encode(images)
            timer.report(f'train batch {train_step} generator encoded')
 
            z = generator.sampling(z_mu, z_sigma)
            timer.report(f'train batch {train_step} generator sampling')
            # Generate random noise
            noise = torch.randn_like(z).to(device)
            timer.report(f'train batch {train_step} noise')

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            timer.report(f'train batch {train_step} timesteps')

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=generator, diffusion_model=unet, noise=noise, timesteps=timesteps
            )
            timer.report(f'train batch {train_step} noise_pred')

            loss = F.mse_loss(noise_pred.float(), noise.float())
            timer.report(f'train batch {train_step} loss')

        scaler_u.scale(loss).backward()
        scaler_u.step(optimizer_u)
        scaler_u.update()
        timer.report(f'train batch {train_step} unet backward')

        epoch_loss += loss.item()
        train_images_seen += len(images)
        recons_loss = epoch_loss / train_images_seen
        print("Epoch [{}] Step [{}/{}] :: recons_loss: {:,.3f}".format(epoch, train_step+1, total_steps, recons_loss))

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size
        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            checkpoint = {
                # Universals
                "args": args,
                "epoch": epoch,
  
                # State variables
                "unet": unet.module.state_dict(),
                "optimizer_u": optimizer_u.state_dict(),
                "scaler_u": scaler_u.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),

                # Evaluation metrics
                "train_images_seen": train_images_seen,
                "val_images_seen": val_images_seen,
                "epoch_loss": epoch_loss,
                "val_loss": val_loss,
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    return unet, timer


def evaluate_diffusion(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
    ):

        unet.eval()

        val_step = val_sampler.progress // val_loader.batch_size
        print(f'\nEvaluating / resuming epoch {epoch} from training step {val_step}\n')

        with torch.no_grad():
            for step, batch in enumerate(val_loader):

                images = batch["image"].to(device)
                timer.report(f'eval batch {val_step} to device')

                with autocast(enabled=True):

                    z_mu, z_sigma = generator.encode(images)
                    timer.report(f'eval batch {val_step} generator encoded')
                    z = generator.sampling(z_mu, z_sigma)
                    timer.report(f'eval batch {val_step} generator sampling')
                    noise = torch.randn_like(z).to(device)
                    timer.report(f'eval batch {val_step} noise')
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    timer.report(f'eval batch {val_step} timesteps')
                    noise_pred = inferer(inputs=images,diffusion_model=unet,noise=noise,timesteps=timesteps,autoencoder_model=generator)
                    timer.report(f'eval batch {val_step} noise_pred')
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    timer.report(f'eval batch {val_step} loss')

                val_loss += loss.item()
                val_images_seen += len(images)
                timer.report(f'eval batch {val_step} metrics update')

                ## Checkpointing
                print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
                val_sampler.advance(len(images))
                if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch
                    print(f"Saving checkpoint at epoch {epoch} train batch {val_step}")
                    checkpoint = {
                        # Universals
                        "args": args,
                        "epoch": epoch,
        
                        # State variables
                        "unet": unet.module.state_dict(),
                        "optimizer_u": optimizer_u.state_dict(),
                        "scaler_u": scaler_u.state_dict(),
                        "train_sampler": train_sampler.state_dict(),
                        "val_sampler": val_sampler.state_dict(),

                        # Evaluation metrics
                        "train_images_seen": train_images_seen,
                        "val_images_seen": val_images_seen,
                        "epoch_loss": epoch_loss,
                        "val_loss": val_loss,
                    }
                    timer = atomic_torch_save(checkpoint, args.resume, timer)

        val_loss /= val_images_seen
        print(f"Epoch {epoch} diff val loss: {val_loss:.4f}")

        return timer
