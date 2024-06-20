import torch
from torch import nn,einsum
from torch.nn import functional as F
from einops import rearrange
import torch.nn.init as init
from math import log2, sqrt

class LogitLaplaceLoss(nn.Module):
    def __init__(self):
        super(LogitLaplaceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Extract the reconstructed values and scale factors from the decoder output
        recon_x = y_pred[:, :3, :, :]  # Reconstructed values
        scales = y_pred[:, 3:, :, :] + 1e-6   # Scale factors

        # Compute the absolute difference between the original input and the reconstructed output
        abs_diff = torch.abs(y_true - recon_x)

        # Compute the Logit Laplace reconstruction loss
        loss = torch.mean(2.0 * scales * (torch.log(2.0 * scales) - torch.log(abs_diff + 1e-10)) + abs_diff / scales)

        return loss

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner
class D_VAE(nn.Module):

    def __init__(self, reconstruction_loss='smooth_l1_loss',
                 logit_laplace_eps=None):
        super(D_VAE, self).__init__()

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),    # 128 ⇒ 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 64 ⇒ 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32 ⇒ 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 8192, 1)

        )
        self.codebook = nn.Embedding(8192, 2048) # this is the code book

        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(2048, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 1)
        )
        self.loss_fn = F.smooth_l1_loss
        self._kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

        if logit_laplace_eps is not None:
            self._recon_loss = LogitLaplaceLoss()
        else:
            if reconstruction_loss == 'smooth_l1_loss':
                self._recon_loss = torch.nn.SmoothL1Loss(reduction='mean')
            elif reconstruction_loss == 'mse_loss':
                self._recon_loss = torch.nn.MSELoss(reduction='mean')
            else:
                raise ValueError(f'Loss {reconstruction_loss} is not supported')
    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def forward(self, x, return_logits=False):
        if self._logit_laplace_eps is not None:
            x = (1 - 2 * self._logit_laplace_eps) * x + self._logit_laplace_eps

        logits = self.conv_e(x)

        if return_logits:
            return logits
        temp = 0.9

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1)

        sampled_info = torch.einsum('b n h w, n d -> b d h w',
                                    soft_one_hot, self.codebook.weight)

        out = self.conv_d(sampled_info)

        mu = out[:, :3, :, :]

        if self._logit_laplace_eps is not None:
            recon_loss = self._recon_loss(x, out)
        else:
            recon_loss = self._recon_loss(x, mu)

        log_prior = torch.log(torch.tensor(1. / 8192))

        # Posterior is p(z|x), which the softmax output of the encoder
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_posterior = F.log_softmax(logits, dim=-1)

        kl_div_loss = self._kl_loss(log_prior, log_posterior)

        overall_loss = recon_loss + 0.01 * kl_div_loss

        if self._logit_laplace_eps is not None:
            reconstructed_img = torch.clamp((mu - self._logit_laplace_eps) / (1 - 2 * self._logit_laplace_eps), 0, 1)
        else:
            reconstructed_img = mu

        return overall_loss, reconstructed_img,recon_loss,kl_div_loss
