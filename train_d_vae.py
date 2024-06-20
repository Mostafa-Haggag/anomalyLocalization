import os
import torch
from torch.nn import functional as F
from dataset import return_MVTecAD_loader
from network import VAE, loss_function, AE, loss_function_2, VAE_new,loss_function_3
from unet import UNetModel_noskipp
from vq_vae import D_VAE
import matplotlib.pyplot as plt
import logging
import wandb
import numpy as np
from tqdm import tqdm
import io
from PIL import Image
from datetime import datetime
import random
from torchvision.transforms import transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def compute_mse_module(recon_batch, data):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(recon_batch, data)
    return mse.item()


def train(model, train_loader, device, optimizer, epoch,loss_type):
    model.train()
    train_loss = 0
    train_kld = 0
    train_reconstruction = 0
    train_mse = 0
    wandb_images_images = []

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            loss, reconstructed_img, recon_loss, kl_div_loss = model(data)
            train_kld += kl_div_loss.item()
            train_mse += compute_mse_module(reconstructed_img, data)
            train_reconstruction += recon_loss.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"Total Loss": loss.item()})
            pbar.update()
            if (batch_idx + 1) == len(train_loader) :
                original_images_plotting = data[:5].detach().cpu().numpy()
                reconstructed_images_plotting = reconstructed_img[:5].detach().cpu().numpy()
                for i, (original, reconstructed) in enumerate(zip(original_images_plotting,
                                                                  reconstructed_images_plotting)):
                    original = (original*255).astype(np.uint8).transpose(1, 2, 0)
                    reconstructed = (reconstructed*255).astype(np.uint8).transpose(1, 2, 0)
                    fig, (
                        orginal_img,
                        reconstructed_img,
                    ) = plt.subplots(
                        nrows=1,
                        ncols=2,
                        figsize=((original.shape[1] * 2) / 96, original.shape[0] / 96),
                        dpi=96,
                    )
                    orginal_img.axis("off")
                    orginal_img.imshow(original)
                    orginal_img.imshow(original)
                    orginal_img.set_title("Org", fontsize=12)
                    reconstructed_img.axis("off")
                    reconstructed_img.imshow(reconstructed, )
                    reconstructed_img.set_title("Reconst", fontsize=12)
                    final_image = fig2img(fig)
                    plt.close(fig)
                    plt.close("all")
                    wandb_images_images.append(wandb.Image(final_image))
    train_loss /= len(train_loader.dataset)
    train_mse /= len(train_loader.dataset)
    train_kld /= len(train_loader.dataset)
    train_reconstruction /= len(train_loader.dataset)
    wandb.log({"train/Paired Images": wandb_images_images,"train/Total Loss": train_loss,
                   "train/RECON":train_reconstruction,"train/KLD":train_kld,"train/mse":train_mse}, step=epoch)



def validation(model, test_loader, device, epoch,loss_type):
    model.eval()
    valid_loss = 0
    valid_kld = 0
    valid_mse = 0
    valid_reconstruction = 0
    wandb_images_images = []
    with tqdm(total=len(test_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            with torch.no_grad():
                loss, reconstructed_img, recon_loss, kl_div_loss = model(data)
                valid_kld += kl_div_loss.item()
                valid_mse += compute_mse_module(reconstructed_img, data)
                valid_reconstruction += recon_loss.item()
                valid_loss += loss.item()
                pbar.set_postfix({"Total Loss": loss.item()})
                pbar.update()
                original_images_plotting = data[5].detach().cpu().numpy()
                reconstructed_images_plotting = reconstructed_img[5].detach().cpu().numpy()
                original = (original_images_plotting*255).astype(np.uint8).transpose(1, 2, 0)
                reconstructed = (reconstructed_images_plotting*255).astype(np.uint8).transpose(1, 2, 0)
                fig, ( orginal_img, reconstructed_img,
                    ) = plt.subplots(
                        nrows=1,
                        ncols=2,
                        figsize=((original.shape[1] * 2) / 96, original.shape[0] / 96),
                        dpi=96,
                    )
                orginal_img.axis("off")
                orginal_img.imshow(original)
                orginal_img.set_title("Org", fontsize=12)
                reconstructed_img.axis("off")
                reconstructed_img.imshow(reconstructed, )
                reconstructed_img.set_title("Reconst", fontsize=12)
                final_image = fig2img(fig)
                plt.close(fig)
                plt.close("all")
                wandb_images_images.append(wandb.Image(final_image))
    valid_loss /= len(test_loader.dataset)
    valid_kld /= len(test_loader.dataset)
    valid_reconstruction /= len(test_loader.dataset)
    wandb.log({"Valid/Paired Images": wandb_images_images,"Valid/Total Loss": valid_loss,
                   "Valid/RECON": valid_reconstruction,"Valid/KLD": valid_kld,
                   "Valid/mse": valid_mse}, step=epoch)
    return valid_loss


def main(config_Dict):
    train_loader = return_MVTecAD_loader(image_dir=config_Dict['training_set'],
                                         batch_size=config_Dict["batch_size"],
                                         image_size=config_Dict["input_size"],
                                         train=True)
    test_loader = return_MVTecAD_loader(image_dir=config_Dict['testing_set'],
                                        batch_size=config_Dict["batch_size"],
                                        image_size=config_Dict["input_size"],
                                        train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = D_VAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Total trainable params: {total_trainable_params:,}")

    wandb.watch(model, log='gradients', log_freq=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=config_Dict["lr"])
    num_epochs = config_Dict["epoch"]
    for epoch in range(num_epochs):
        print("Starting Training for epoch {}".format(epoch + 1))
        loss = train(model=model, train_loader=train_loader, device=device, optimizer=optimizer, epoch=epoch,
                     loss_type=config_Dict["loss_type"])
        print("Starting Testing for epoch {}".format(epoch + 1))
        loss_valid = validation(model=model, test_loader=test_loader, device=device, epoch=epoch,
                                loss_type=config_Dict["loss_type"])
        print("Average losses for epoch {}".format(epoch + 1))
        print('epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1, num_epochs, loss))
        print('epoch [{}/{}], validation loss: {:.4f}'.format(epoch + 1, num_epochs, loss_valid))
        if (epoch + 1) % config_Dict["saving_epoch"] == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "{}.pth".format(epoch + 1)))


if __name__ == "__main__":
    run_id = wandb.util.generate_id()
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    print(f"Starting a new wandb run with id {run_id}")
    config_dict = {"batch_size": 16,
                   "epoch": 20,
                   "lr": 5e-4,
                   "z_dim": 512,
                   "model_id": 'UNET_DESIGN',
                   "tag": "u_net_no_skip_connections",
                   "unet_inout_channels": 3,
                   "unet_inplanes": 16,
                   "unet_residual": 1,
                   "unet_attention": [8],
                   "unet_dropout": 0.0,
                   "unet_mult": [1,2,3,4],
                   "unet_resample": False,
                   "unet_dims": 2,
                   "unet_checkpoint": False,
                   "unet_heads": 4,
                   "unet_num_heads_channels": 4,
                   "unet_res_updown": False,
                   "saving_epoch": 1,
                   "loss_type": "mse",
                   "input_size": 224,
                   "training_set": r"D:\github_directories\foriegn\black_plate_c_shape_blue_pills\SEQ00003_MIXED_COUNTED_313",
                   "testing_set": r"D:\github_directories\foriegn\black_plate_c_shape_blue_pills\SEQ00004_MIXED_FOREIGN_PARTICLE"
                   }

    wandb.init(
        project="Foreign_Particle_project_organised",
        config=config_dict,
        tags=["u_net", config_dict["tag"]],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'{timestamp}_{config_dict["model_id"]}_{config_dict["tag"]}'
    wandb.run.name = new_run_name

    # Output directory with timestamp
    out_dir = os.path.join('.', 'logs', new_run_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f'Created directory: {out_dir}')
    else:
        logger.info(f'Directory already exists: {out_dir}')

    # Checkpoints directory with timestamp
    checkpoints_dir = os.path.join('.', 'checkpoints', new_run_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        logger.info(f'Created directory: {checkpoints_dir}')
    else:
        logger.info(f'Directory already exists: {checkpoints_dir}')

    main(config_dict)