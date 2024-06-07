import torch
from torch import nn
from torch.nn import functional as F
class VAE(nn.Module):

    def __init__(self, z_dim=128):
        super(VAE, self).__init__()

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # 512 ⇒ 256
            nn.BatchNorm2d(32),            
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 256 ⇒ 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 ⇒ 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim*2),
        )

        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 16 * 16),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.z_dim = z_dim
        self.initialize()

    def encode(self, input):
        # input is greyscale (1,128,128)
        x = self.conv_e(input) # the size is torch.Size([256, 128, 16, 16])
        x = x.view(-1, 128*16*16)# you flatten everything
        # [256, 32768]

        x = self.fc_e(x)# [256,1024] returned size
        # you divide them depedning on z_dim
        # it is as if oyu have two parts
            # mu: The mean vector of the latent Gaussian distribution.
            # logvar: The logarithm of the variance vector of the latent Gaussian distribution.
                # Using the logarithm of the variance helps maintain numerical stability during training.
        return x[:, :self.z_dim], x[:, self.z_dim:]

    def reparameterize(self, mu, logvar):
        # The reparameterize function uses the reparameterization trick to generate samples from a Gaussian
        # distribution with mean mu and variance exp(logvar)
        # During training, it introduces randomness by sampling from a standard normal distribution
        # and scaling/shifting the samples.
        # : This condition checks if the model is currently in training mode.
        # If it is, the reparameterization trick is applied. If not (e.g., during evaluation or inference),
        # the mean vector mu is directly returned.
        if self.training:

            # Multiply logvar by 0.5. Take the exponential to obtain the standard deviation.
            # simple math is happening in here
            std = logvar.mul(0.5).exp_()
            # create a new tensor eps with the same size as std, filled with samples from a standard normal distribution
            # (mean 0 and variance 1).
            eps = std.new(std.size()).normal_()

            # This performs the reparameterization
            # eps.mul(std): Scale the standard normal samples (eps) by the standard deviation (std).
            #  Shift the scaled samples by the mean (mu). This effectively samples from the Gaussian distribution
            #  you shift the normal distribtuion by mean and multiple by std
            return eps.mul(std).add_(mu)
        else:
            return mu
            # If the model is not in training mode (e.g., during evaluation), the mean vector mu is returned directly
            # without adding any noise. This ensures deterministic behavior during inference.

    def decode(self, z):
        # you have an input of [256,512]
        # torch.Size([128, 512])
        h = self.fc_d(z)# full convelutions
        h = h.view(-1, 128, 16, 16)
        return self.conv_d(h)# unsampling

    def initialize(self):
        # this is very important
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) # z should have examly same size as mu and logvar  [batchsize,512]
        self.mu = mu
        self.logvar = logvar
        return self.decode(z)

'''
The loss_function works as follows:

Reconstruction Loss: Measures how well the VAE can reconstruct the input data.
KL Divergence Loss: Regularizes the latent space to follow a standard normal distribution.
Total Loss: Sum of the reconstruction loss and KL divergence loss, balancing the trade-off between reconstruction accuracy and latent space regularization.

'''
## The loss_function in a Variational Autoencoder (VAE) combines two components:
# the reconstruction loss and the Kullback-Leibler divergence (KL divergence) loss.
    # This combination ensures that the VAE can reconstruct the input data accurately
    # while also regularizing the latent space to follow a standard normal distribution.
# Components of the Loss Function
 # Reconstruction Loss:     recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
 # Purpose: Measures how well the reconstructed data (recon_x) matches the original input data (x).
 # : Sums the loss over all elements, giving a single scalar value representing the total reconstruction error for the entire batch.
def loss_function(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Regularizes the latent space by ensuring that the distribution
    # of the encoded latent vectors (mu and logvar) approximates a standard normal distribution (mean = 0, variance = 1).
    return recon + kld



