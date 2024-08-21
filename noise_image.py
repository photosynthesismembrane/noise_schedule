import torchvision.transforms as transforms
import torch
import os
from torchvision.utils import save_image
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Example usage:
# py noise_image.py --image_path van_gogh.png --output_folder cosine_noised_images --schedule cosine --T 1001 --save_every_n_steps 100 --center_crop 899

class GaussianDiffusion:
    """
    A class for Gaussian diffusion.

    This class provides methods for sampling from the diffusion process and
    computing the noise schedule.
    
    Code adapted from:
    https://github.com/CompVis/latent-diffusion
    """

    def __init__(self, schedule, T):
        """
        Create a Gaussian diffusion process.

        :param schedule: the name of the beta schedule to use.
        :param T: the number of diffusion steps to use.
        """
        # Check for GPU and set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.T = T
        # Use float64 for accuracy.
        self.betas = np.array(self.get_named_beta_schedule(schedule, self.T), dtype=np.float64)

        # Define the noise schedule (linearly increasing from beta_1 to beta_T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)


    def get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps):
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif schedule_name == "cosine":
            return self.betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def q_sample(self, x_start, t, noise=None):
            """
            Diffuse the data for a given number of diffusion steps.

            In other words, sample from q(x_t | x_0).

            :param x_start: the initial data batch.
            :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
            :param noise: if specified, the split-out normal noise.
            :return: A noisy version of x_start.
            """
            if noise is None:
                noise = torch.randn_like(x_start)
            assert noise.shape == x_start.shape
            return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )


    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=self.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


# Read arguments from the command line
parser = argparse.ArgumentParser(description='Noise an image using Gaussian diffusion.')
parser.add_argument('--image_path', type=str, default='image.png', help='Path to the image to noise.')
parser.add_argument('--output_folder', type=str, default='noised_images', help='Folder to save the noised images.')
parser.add_argument('--schedule', type=str, default='linear', help='The beta schedule to use.')
parser.add_argument('--T', type=int, default=1001, help='The number of diffusion steps.')
parser.add_argument('--save_every_n_steps', type=int, default=100, help='Save an image every n steps.')
parser.add_argument('--center_crop', type=int, default=0, help='Size of the center crop.')
args = parser.parse_args()


# Directory for saving images
save_dir = args.output_folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Center crop image to be square
transform = transforms.Compose([transforms.ToTensor()])
if args.center_crop > 0:
    transform = transforms.Compose([transforms.CenterCrop(args.center_crop), transforms.ToTensor()])

# Load an image from a file
image_path = args.image_path
image = Image.open(image_path)

# Apply the transform
image = transform(image)

# Move the image to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
image = image.to(device)
image = (image - .5) * 2.


diffusion_cosine = GaussianDiffusion(args.schedule, args.T)
number_of_steps = math.ceil(args.T / args.save_every_n_steps)

fig, axes = plt.subplots(nrows=1, ncols=number_of_steps, figsize=(20, 3))

for i, t in enumerate(range(0, args.T, args.save_every_n_steps)):

    noised_image = diffusion_cosine.q_sample(image, t)

    # Normalize the image to be in the range [0, 1]
    noised_image = ((noised_image / 2.) + .5)

    # Clip image values between [0, 1]
    noised_image = noised_image.clamp(0, 1)

    # Save the image
    save_image(noised_image, os.path.join(save_dir, f'noised_image_{t}.png'))

    # Plot the image on the ith subplot
    ax = axes[i]
    ax.imshow(noised_image.permute(1, 2, 0).cpu().numpy())
    ax.set_title(f't={t}')
    ax.axis('off')  # Remove axis ticks and labels


fig.suptitle(args.schedule.capitalize(), fontsize=16)

plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
plt.savefig(os.path.join(save_dir, f'{args.schedule}_diffusion_forward.png'))