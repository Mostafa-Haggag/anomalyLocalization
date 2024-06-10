import os
import torch
from torch.nn import functional as F
from dataset import return_MVTecAD_loader
from network import VAE,loss_function , AE , loss_function_2
import matplotlib.pyplot as plt
import logging
import wandb
import numpy as np
from tqdm import tqdm
import io
from PIL import Image
from datetime import datetime
import random
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_images(original, reconstructed):
    # Convert PyTorch tensors to NumPy arrays and reshape to (batch_size, height, width)
    original_images = original[:5].squeeze(1).detach().cpu().numpy()
    reconstructed_images = reconstructed[:5].squeeze(1).detach().cpu().numpy()
    # Create a grid to combine original and reconstructed images
    num_images = original_images.shape[0]

    combined_grid = np.zeros((num_images,2,original_images.shape[1],original_images.shape[2]))
    # print(combined_grid.shape)
    # print(original_images.shape)

    combined_grid[:, 0, :, :] = original_images
    combined_grid[:, 1, :, :] = reconstructed_images

    # Concatenate images along the width dimension to create a grid
    ##grid_image = np.concatenate(combined_grid, axis=3)
    ##grid_image = np.concatenate(combined_grid, axis=1)

    # Convert the grid image back to PyTorch tensor
    return combined_grid
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def train(model, train_loader, device, optimizer, epoch):
    model.train()
    train_loss = 0
    log_interval = 1  # Adjust this value as needed
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data)
            #loss = loss_function_2(recon_batch, data)
            loss = loss_function(recon_batch, data, model.mu, model.logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"Reconstruction Loss": loss.item()})
            pbar.update()
            if (batch_idx + 1) == len(train_loader) :
                original_images_plotting = data[:5].detach().cpu().numpy()
                reconstructed_images_plotting = recon_batch[:5].detach().cpu().numpy()
                wandb_images_images = []
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
                    orginal_img.imshow(original, cmap='gray')
                    #orginal_img.imshow(original)

                    orginal_img.set_title("Org", fontsize=12)
                    reconstructed_img.axis("off")
                    reconstructed_img.imshow(reconstructed, cmap='gray')
                    reconstructed_img.set_title("Reconst", fontsize=12)
                    final_image = fig2img(fig)
                    ##final_image.save(f"comparison_{i}.png")
                    #plt.show()
                    plt.close(fig)
                    plt.close("all")
                    wandb_images_images.append(wandb.Image(final_image))
    train_loss /= len(train_loader.dataset)
    wandb.log({"Paired Images":wandb_images_images,"Reconstruction Loss over batch": train_loss})
    return train_loss


def eval(model,test_loader,device):
    model.eval()
    x_0 = iter(test_loader).next()
    with torch.no_grad():
        x_vae = model(x_0.to(device)).detach().cpu().numpy()

# for testing
# what is happening in here ?
# Your EBM (Energy-Based Model) function seems to be designed for testing a given model
# by performing iterative gradient-based updates on the input data x_0
def EBM(model,test_loader,device):
    model.train()
    x_0 = iter(test_loader).next()
    alpha = 0.05
    lamda = 1
    # alpha = 0.05 and lamda = 1: Hyperparameters for the gradient update steps.
    x_0 = x_0.to(device).clone().detach().requires_grad_(True)
    # Moves the batch to the specified device, clones it, detaches it from the computation graph, and enables gradient computation.
    recon_x = model(x_0).detach()
    # Passes x_0 through the model and detaches the output to avoid computing gradients for this part of the graph.
    loss = F.binary_cross_entropy(x_0, recon_x, reduction='sum')
    #  Computes the reconstruction loss between the input and the reconstructed output.
    loss.backward(retain_graph=True)
    #  Computes the gradients of the loss with respect to x_0.

    x_grad = x_0.grad.data # : Retrieves the gradients of x_0
    x_t = x_0 - alpha * x_grad * (x_0 - recon_x) ** 2
    #  Updates x_0 based on the gradients and the difference between x_0 and the reconstruction.
    for i in range(15):
        # A loop of 15 iterations is used to refine x_t
        recon_x = model(x_t).detach()
        loss = F.binary_cross_entropy(x_t, recon_x, reduction='sum') + lamda * torch.abs(x_t - x_0).sum()
        loss.backward(retain_graph=True)

        x_grad = x_0.grad.data
        eps = 0.001
        # there is no alpha
        x_t = x_t - eps * x_grad * (x_t - recon_x) ** 2
        iterative_plot(x_t.detach().cpu().numpy(), i)

        
# gif
def iterative_plot(x_t, j):
    plt.figure(figsize=(15, 4))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_t[i][0], cmap='gray')
    plt.subplots_adjust(wspace=0., hspace=0.)        
    plt.savefig("./results/{}.png".format(j))
    #plt.show()
# def plot_reconstructed_images(model, data_loader, num_images=10):
#     model.eval()
#
#     with tqdm(total=len(data_loader), unit="batch") as pbar:
#         for batch_idx, data in enumerate(data_loader):
#             with torch.no_grad():
#                 recon_images, = model(data)
#
#                 fig, axes = plt.subplots(nrows=2, ncols=num_images, sharex=True, sharey=True, figsize=(15, 4))
#
#                 for images, row in zip([data, recon_images], axes):
#                     for img, ax in zip(images, row):
#                         ax.imshow(img.cpu().numpy().squeeze(), cmap='gray')
#                         ax.get_xaxis().set_visible(False)
#                         ax.get_yaxis().set_visible(False)


def main(config_Dict):
    train_loader = return_MVTecAD_loader(image_dir=r"D:\github_directories\foriegn\SEQ00003_MIXED_COUNTED_313", batch_size=config_Dict["batch_size"], train=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Output directory
    # out_dir = os.path.join('.', 'logs')
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    #     logger.info(f'Created directory: {out_dir}')
    # else:
    #     logger.info(f'Directory already exists: {out_dir}')
    #
    # # Checkpoints directory
    # checkpoints_dir = os.path.join('.', 'checkpoints')
    # if not os.path.exists(checkpoints_dir):
    #     os.mkdir(checkpoints_dir)
    #     logger.info(f'Created directory: {checkpoints_dir}')
    # else:
    #     logger.info(f'Directory already exists: {checkpoints_dir}')
    # Generate a timestamp
        # out_dir = '.logs'
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # checkpoints_dir =".checkpoints"
    # if not os.path.exists(checkpoints_dir):
    #     os.mkdir(out_dir)

    model = VAE(z_dim=config_Dict["z_dim"]).to(device)
    #model = AE(latent_size=config_dict["z_dim"],img_size=256,vae=False).to(device)

    ##wandb.watch(model, log='gradients',log_freq=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=config_Dict["lr"])
    num_epochs = config_Dict["epoch"]
    for epoch in range(num_epochs):
        loss = train(model=model,train_loader=train_loader,device=device,optimizer=optimizer,epoch=epoch)
        print('epoch [{}/{}], train loss: {:.4f}'.format(epoch + 1,num_epochs,loss))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir,"{}.pth".format(epoch+1)))

    test_loader = return_MVTecAD_loader(image_dir=r"D:\github_directories\foriegn\SEQ00004_MIXED_FOREIGN_PARTICLE", batch_size=10, train=False)
    ##plot_reconstructed_images(model, test_loader)
    ##plt.show()
    
if __name__ == "__main__":
    run_id = wandb.util.generate_id()
    seed =1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    print(f"Starting a new wandb run with id {run_id}")
    config_dict = {"batch_size": 128,
                   "epoch": 40,
                   "lr": 1e-5,
                   "z_dim": 512,
                   "model_id":'vae_reconstruction',
                   "tag":"normal_runs"
                   }

    wandb.init(
        # set the wandb project where this run will be logged
        project="Foreign_Particle_project",
        config=config_dict,
        tags=["VAE_understanding", config_dict["tag"]],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name =f'{timestamp}_{config_dict["model_id"]}_{config_dict["tag"]}'
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