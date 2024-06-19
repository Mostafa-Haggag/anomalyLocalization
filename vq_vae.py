import torch
from torch import nn,einsum
from torch.nn import functional as F
from einops import rearrange
import torch.nn.init as init
from math import log2, sqrt

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner
class VAE(nn.Module):

    def __init__(self, z_dim=128):
        super(VAE, self).__init__()

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

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices
    def decode(
            self,
            img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(
            image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.fc_d(z)
        h = h.view(-1, 128, 16, 16)
        return self.conv_d(h)

    def forward(self, x):
        logits = self.conv_e(x)
        temp = 0.9
        soft_one_hot = F.gumbel_softmax(
            logits, tau=temp, dim=1, hard=False)
        sampled = einsum('b n h w, n d -> b d h w',
                         soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)
        recon_loss = self.loss_fn(x, out)
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / 8192], device=x.device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None,
                          'batchmean', log_target=True)
        loss = recon_loss + (kl_div * 0.01)
        return loss, out
