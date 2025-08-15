import torch
import torchvision
import torch.nn.functional as F
from torch import nn, autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import model as ebm
import dataloader as dl
import argparse


class DiffusionHelper:
    """Manages the diffusion noise schedule according to the paper."""
    def __init__(self, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # [cite_start]The paper uses a linear schedule for sigma^2 [cite: 202]
        # We define sigmas, which are the std dev of the added noise
        self.sigmas_squared = torch.linspace(1e-4, 0.4, num_timesteps, device=device)
        self.sigmas = torch.sqrt(self.sigmas_squared)
        
        # [cite_start]The scaling factor for the forward process [cite: 170]
        self.sqrt_one_minus_sigma_squared = torch.sqrt(1. - self.sigmas_squared)

    def get_data_pair(self, x_0, t):
        """
        [cite_start]Generates the pair (y_t, x_{t+1}) from a clean image x_0. [cite: 185]
        """
        # Noise x_0 to get x_t
        noise_t = torch.randn_like(x_0)
        x_t = self.sqrt_one_minus_sigma_squared[t].view(-1, 1, 1, 1) * x_0 + self.sigmas[t].view(-1, 1, 1, 1) * noise_t

        # Noise x_t to get x_{t+1}
        noise_t1 = torch.randn_like(x_t)
        # Note: The paper's notation x_{t+1} = sqrt(1-sigma_{t+1}^2)x_t + ... can be confusing.
        # Here we use the sigma for the transition from t -> t+1.
        # Let's use the sigma at step t to keep it consistent with the model's conditioning.
        x_t1 = self.sqrt_one_minus_sigma_squared[t].view(-1, 1, 1, 1) * x_t + self.sigmas[t].view(-1, 1, 1, 1) * noise_t1

        # [cite_start]y_t is a scaled version of x_t [cite: 171]
        y_t = self.sqrt_one_minus_sigma_squared[t].view(-1, 1, 1, 1) * x_t

        return y_t, x_t1



def run_conditional_mcmc(model, y_t_initial, x_t1_condition, t, K, b, sigma_t):
    """
    Runs K steps of conditional Langevin MCMC to sample from p(y_t | x_{t+1}).
    [cite_start]This directly implements Equation 17. [cite: 177, 187, 195]
    """
    y_k = y_t_initial.detach().clone().requires_grad_(True)
    step_size_sq = (b * sigma_t)**2

    for _ in range(K):
        # Get the energy and the score (gradient) of the UNCONDITIONAL EBM f(y, t)
        energy = model(y_k, t)
        score_unconditional = autograd.grad(energy.sum(), y_k, retain_graph=False)[0]

        # The gradient of the full conditional log-likelihood includes a second term
        # [cite_start]from the quadratic part of the conditional EBM [cite: 122, 177]
        score_conditional_term = (1 / sigma_t**2) * (x_t1_condition - y_k)
        
        # The full gradient for the conditional distribution p(y_t | x_{t+1})
        full_score = -score_unconditional + score_conditional_term

        # [cite_start]Langevin dynamics update step [cite: 177]
        noise = torch.randn_like(y_k)
        y_k = y_k + 0.5 * step_size_sq * full_score + torch.sqrt(step_size_sq) * noise
        y_k = y_k.detach().clone().requires_grad_(True)

    return y_k.detach()



@torch.no_grad()
def progressive_sample(model, diffusion_helper, n_samples=16, K=30, b=0.1, device='cpu'):
    """
    [cite_start]Implements the progressive sampling procedure from Algorithm 2. [cite: 191]
    """
    model.to(device)
    model.eval()

    # [cite_start]1. Start with pure Gaussian noise [cite: 192]
    x_t = torch.randn(n_samples, 3, 32, 32, device=device)

    # [cite_start]2. Loop backwards from T-1 to 0 [cite: 192]
    for t_int in tqdm(reversed(range(diffusion_helper.num_timesteps)), total=diffusion_helper.num_timesteps, desc="Sampling"):
        t = torch.full((n_samples,), t_int, device=device, dtype=torch.long)
        x_t1_condition = x_t

        # [cite_start]3. Initialize y_t and refine with K steps of MCMC [cite: 193, 194, 195]
        y_t_initial = x_t1_condition.clone()
        sigma_t_batch = diffusion_helper.sigmas[t].view(-1, 1, 1, 1)

        # We must re-enable gradients for the MCMC run even inside no_grad() context
        with torch.enable_grad():
            y_t_final = run_conditional_mcmc(
                model, y_t_initial, x_t1_condition, t, K, b, sigma_t_batch
            )

        # [cite_start]4. Rescale to get the new x_t [cite: 196]
        scaling_factor = diffusion_helper.sqrt_one_minus_sigma_squared[t].view(-1, 1, 1, 1)
        x_t = y_t_final / scaling_factor

    # Denormalize from [-1, 1] to [0, 1] for saving
    final_samples = (x_t.clamp(-1, 1) + 1) / 2
    return final_samples



def main(args):
    # --- Hyperparameters ---
    LOAD_MODEL = args.load_model
    SAVE_FILE = args.save_file
    NUM_TIMESTEPS = args.num_timesteps  # T in the paper [cite: 200]
    MCMC_K = args.mcmc_k                # Number of MCMC steps [cite: 199]
    MCMC_B = args.mcmc_b                # MCMC step size hyperparameter [cite: 175]

    # --- Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    model = ebm.EBM_Unet()
    diffusion_helper = DiffusionHelper(num_timesteps=NUM_TIMESTEPS, device=DEVICE)
    
    state_dict = torch.load(LOAD_MODEL, map_location=DEVICE)
    print(f"Successfully loaded state dictionary from '{LOAD_MODEL}'.")

    # Now, load the state dictionary into the model instance.
    model.load_state_dict(state_dict)
    print("Weights and parameters applied to the model.")
    
    # --- Inference ---
    print("\n--- Starting Inference ---")
    generated_samples = progressive_sample(
        model, diffusion_helper, n_samples=16, K=MCMC_K, b=MCMC_B, device=DEVICE
    )
    # Save the generated images
    torchvision.utils.save_image(generated_samples, SAVE_FILE, nrow=4)
    print(f"Generated samples saved to {SAVE_FILE}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--load_model",      type=str,   default="/content/drive/MyDrive/ebm_runs/ebm_recovery_epoch_5.pth")
    p.add_argument("--save_file",       type=str,   default="/content/drive/MyDrive/ebm_runs/recovery_likelihood_samples.png")
    p.add_argument("--num_timesteps",   type=int,   default=500)
    p.add_argument("--mcmc_k",          type=int,   default=40)
    p.add_argument("--mcmc_b",          type=float, default=0.1)
    
    args = p.parse_args()
    main(args)