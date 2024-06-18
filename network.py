import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, latent_size, multiplier=4, img_size=64, vae=False):
        super(AE, self).__init__()
        self.fm = 4 # what is this 4 in here ?
        # what is it indicating ?
        self.mp = multiplier
        # not working in grey scale in here
        # you are passing am ultiplied of 1
        # this 2d conv is halfing  every time
        self.encoder = nn.Sequential(
            nn.Conv2d(3, int(16 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(16 * multiplier),
                      int(32 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(32 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            # 128 x 128
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False) if img_size>=128 else nn.Identity(),
            nn.BatchNorm2d(int(64 * multiplier)) if img_size>=128 else nn.Identity(),
            nn.ReLU(True) if img_size>=128 else nn.Identity(),
            # 256 x 256
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False) if img_size>=256 else nn.Identity(),
            nn.BatchNorm2d(int(64 * multiplier)) if img_size>=256 else nn.Identity(),
            nn.ReLU(True) if img_size>=256 else nn.Identity(),
        )
        if not vae:
            # fm is 4 by 4 indicating that i should have by the end (64,4,4) so that it work
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm*self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size),
            )
            # difference in here that i am mtulpliing the latent size by 2
        else:
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm*self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size * 2),
            )

        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, int(64 * multiplier) * self.fm*self.fm),
        )
        self.decoder = nn.Sequential(
            # 128 x 128
            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False) if img_size>=128 else nn.Identity(),
            nn.BatchNorm2d(int(64*multiplier)) if img_size>=128 else nn.Identity(),
            nn.ReLU(True) if img_size>=128 else nn.Identity(),
            # 256 x 256
            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False) if img_size>=256 else nn.Identity(),
            nn.BatchNorm2d(int(64*multiplier)) if img_size>=256 else nn.Identity(),
            nn.ReLU(True) if img_size>=256 else nn.Identity(),

            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(64*multiplier), int(32 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(32*multiplier), int(16 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(16*multiplier),
                               3, 4, 2, 1, bias=False),
        )
        self.initialize()

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
        lat_rep = self.feature(x)
        out = self.decode(lat_rep)
        return out

    def feature(self, x):
        lat_rep = self.encoder(x)
        lat_rep = lat_rep.view(lat_rep.size(0), -1)
        lat_rep = self.linear_enc(lat_rep)
        return lat_rep

    def decode(self, x):
        out = self.linear_dec(x)
        out = out.view(out.size(0), int(64 * self.mp), self.fm, self.fm)
        out = self.decoder(out)
        out = torch.tanh(out)
        return out


class VAE_new(nn.Module):
    def __init__(self, z_dim=128):
        super(VAE_new, self).__init__()

        # Encode
        self.conv_e1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 224 -> 112
        self.bn_e1 = nn.BatchNorm2d(32)
        self.conv_e2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 112 -> 56
        self.bn_e2 = nn.BatchNorm2d(64)
        self.conv_e3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 56 -> 28
        self.bn_e3 = nn.BatchNorm2d(128)
        self.conv_e4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.bn_e4 = nn.BatchNorm2d(256)
        # self.conv_e5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        # self.bn_e5 = nn.BatchNorm2d(512)

        self.fc_e1 = nn.Linear(256 * 7 * 7, 1024)
        self.bn_e6 = nn.BatchNorm1d(1024)
        self.fc_e2 = nn.Linear(1024, z_dim * 2)

        # Decode
        self.fc_d1 = nn.Linear(z_dim, 1024)
        self.bn_d1 = nn.BatchNorm1d(1024)
        self.fc_d2 = nn.Linear(1024, 256 * 7 * 7)
        self.bn_d2 = nn.BatchNorm1d(256 * 7 * 7)

        self.conv_d2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn_d4 = nn.BatchNorm2d(128)
        self.conv_d3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn_d5 = nn.BatchNorm2d(64)
        self.conv_d4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_d6 = nn.BatchNorm2d(32)
        self.conv_d5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.z_dim = z_dim
        self.initialize()

    def encode(self, input):
        x1 = F.leaky_relu(self.bn_e1(self.conv_e1(input)), 0.2)
        x2 = F.leaky_relu(self.bn_e2(self.conv_e2(x1)), 0.2)
        x3 = F.leaky_relu(self.bn_e3(self.conv_e3(x2)), 0.2)
        x4 = F.leaky_relu(self.bn_e4(self.conv_e4(x3)), 0.2)

        x = x4.view(-1, 256 * 7 * 7)
        x = F.leaky_relu(self.bn_e6(self.fc_e1(x)), 0.2)
        x = self.fc_e2(x)

        return x[:, :self.z_dim], x[:, self.z_dim:], x1, x2, x3, x4

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, x1, x2, x3, x4):
        h = F.leaky_relu(self.bn_d1(self.fc_d1(z)), 0.2)
        h = F.leaky_relu(self.bn_d2(self.fc_d2(h)), 0.2)
        h = h.view(-1, 256, 7, 7)
        h = F.leaky_relu(self.bn_d4(self.conv_d2(h + x4)), 0.2)
        h = F.leaky_relu(self.bn_d5(self.conv_d3(h + x3)), 0.2)
        h = F.leaky_relu(self.bn_d6(self.conv_d4(h + x2)), 0.2)
        out = torch.sigmoid(self.conv_d5(h + x1))

        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mu, logvar, x1, x2, x3, x4 = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decode(z, x1, x2, x3, x4)

## this is for smaller size image
# class VAE_new(nn.Module):
#     def __init__(self, z_dim=128):
#         super(VAE_new, self).__init__()
#
#         # Encode
#         self.conv_e1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
#         self.bn_e1 = nn.BatchNorm2d(32)
#         self.conv_e2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.bn_e2 = nn.BatchNorm2d(64)
#         self.conv_e3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.bn_e3 = nn.BatchNorm2d(128)
#
#         self.fc_e1 = nn.Linear(128 * 16 * 16, 1024)
#         self.bn_e4 = nn.BatchNorm1d(1024)
#         self.fc_e2 = nn.Linear(1024, z_dim * 2)
#
#         # Decode
#         self.fc_d1 = nn.Linear(z_dim, 1024)
#         self.bn_d1 = nn.BatchNorm1d(1024)
#         self.fc_d2 = nn.Linear(1024, 128 * 16 * 16)
#         self.bn_d2 = nn.BatchNorm1d(128 * 16 * 16)
#
#         self.conv_d1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.bn_d3 = nn.BatchNorm2d(64)
#         self.conv_d2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.bn_d4 = nn.BatchNorm2d(32)
#         self.conv_d3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
#
#         self.z_dim = z_dim
#         self.initialize()
#
#     def encode(self, input):
#         x1 = F.leaky_relu(self.bn_e1(self.conv_e1(input)), 0.2)
#         x2 = F.leaky_relu(self.bn_e2(self.conv_e2(x1)), 0.2)
#         x3 = F.leaky_relu(self.bn_e3(self.conv_e3(x2)), 0.2)
#
#         x = x3.view(-1, 128 * 16 * 16)
#         x = F.leaky_relu(self.bn_e4(self.fc_e1(x)), 0.2)
#         x = self.fc_e2(x)
#
#         return x[:, :self.z_dim], x[:, self.z_dim:], x1, x2, x3
#
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def decode(self, z, x1, x2, x3):
#         h = F.leaky_relu(self.bn_d1(self.fc_d1(z)), 0.2)
#         h = F.leaky_relu(self.bn_d2(self.fc_d2(h)), 0.2)
#         h = h.view(-1, 128, 16, 16)
#
#         h = F.leaky_relu(self.bn_d3(self.conv_d1(h + x3)), 0.2)
#         h = F.leaky_relu(self.bn_d4(self.conv_d2(h + x2)), 0.2)
#         out = torch.sigmoid(self.conv_d3(h + x1))
#
#         return out
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         mu, logvar, x1, x2, x3 = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         self.mu = mu
#         self.logvar = logvar
#         return self.decode(z, x1, x2, x3)


class VAE(nn.Module):
    def __init__(self, z_dim=128,input_image_size=64,number_of_channels =3):
        super(VAE, self).__init__()
        self.hidden_dims = [32, 64, 128]
        modules = []
        in_channels = number_of_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2))
            )
            in_channels = h_dim
        self .low_size = input_image_size //(2**len(self.hidden_dims))
        self.conv_e = nn.Sequential(*modules)
        # encode
        # the size by the end of here is 6  by 6
        self.fc_e = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] * self .low_size * self .low_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, z_dim*2),
        )
        self.hidden_dims.reverse()

        # Build Decoder
        modules = []
        for i in range(len(self.hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(self.hidden_dims[-1],
                                       number_of_channels,
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                        nn.Sigmoid()
                    )
                    )

        self.conv_d = nn.Sequential(*modules)

        # decode
        self.fc_d = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.hidden_dims[0] * self .low_size * self .low_size),
            nn.LeakyReLU(0.2)
        )
        self.z_dim = z_dim
        self.initialize()

    def encode(self, input):
        # input is greyscale (1,128,128)
        x = self.conv_e(input) # the size is torch.Size([256, 128, 16, 16])
        x = torch.flatten(x,start_dim=1)

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
        h = h.view(-1, self.hidden_dims[0], self.low_size, self.low_size)
        return self.conv_d(h)# unsampling

    def initialize(self):
        # this is very important
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
    return recon , kld


def loss_function_2(recon_x, x):
    rec_err = (recon_x - x) ** 2
    recon = rec_err.mean()
    return recon

