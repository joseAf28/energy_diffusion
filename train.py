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

# Assume the EBM_Unet model architecture from the previous steps is defined here.
# (If not, you'll need to include the EBM_Unet class definition)

class DiffusionHelper:
    """Manages the diffusion noise schedule according to the paper."""
    def __init__(self, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # [cite_start]The paper uses a linear schedule for sigma^2 [cite: 202]
        # We define sigmas, which are the std dev of the added noise
        self.sigmas_squared = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
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


def train_recovery_likelihood(model, train_loader, optimizer, diffusion_helper,
                            epochs=50, K=30, b=0.1, device='cpu', save_file=""):
    """
    [cite_start]Implements the training procedure from Algorithm 1. [cite: 183]
    """
    model.to(device)
    
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for (clean_images, _) in progress_bar:
            optimizer.zero_grad()
            clean_images = clean_images.to(device)
            batch_size = clean_images.shape[0]

            # [cite_start]1. Sample t and the data pair (y_t, x_{t+1}) [cite: 185]
            t = torch.randint(0, diffusion_helper.num_timesteps, (batch_size,), device=device).long()
            y_t_positive, x_t1_condition = diffusion_helper.get_data_pair(clean_images, t)

            # [cite_start]2. Initialize negative samples from the conditioning variable [cite: 186]
            y_t_negative_initial = x_t1_condition.clone()
            
            # [cite_start]3. Refine negative samples with K steps of MCMC [cite: 187]
            sigma_t_batch = diffusion_helper.sigmas[t].view(-1, 1, 1, 1)
            y_t_negative_final = run_conditional_mcmc(
                model, y_t_negative_initial, x_t1_condition, t, K, b, sigma_t_batch
            )

            # 4. Calculate energies for positive and negative samples
            energy_positive = model(y_t_positive, t)
            energy_negative = model(y_t_negative_final, t)

            # [cite_start]5. Calculate loss and update parameters [cite: 188]
            # The objective is to lower the energy of positive (real) samples
            # and raise the energy of negative (fake) samples.
            loss = energy_positive.mean() - energy_negative.mean()

            # Optional: Add regularization to prevent energy values from exploding
            reg_loss = 0.01 * ((energy_positive**2).mean() + (energy_negative**2).mean())
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            
        
        model_path = save_file + f"ebm_recovery_epoch_{epoch+1}.pth"
        # Save a checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1}")
        elif best_loss < total_loss:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1}")
        
        best_loss = total_loss if total_loss < best_loss else best_loss
        

def main(args):
    # --- Hyperparameters ---
    save_file = args.save_file
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_TIMESTEPS = args.num_timesteps  # T in the paper [cite: 200]
    MCMC_K = args.mcmc_k                # Number of MCMC steps [cite: 199]
    MCMC_B = args.mcmc_b                # MCMC step size hyperparameter [cite: 175]

    # --- Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    model = ebm.EBM_Unet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, _ = dl.get_cifar10_dataloaders(batch_size=BATCH_SIZE)
    diffusion_helper = DiffusionHelper(num_timesteps=NUM_TIMESTEPS, device=DEVICE)

    # --- Training ---
    print("--- Starting Training ---")
    train_recovery_likelihood(
        model, train_loader, optimizer, diffusion_helper,
        epochs=EPOCHS, K=MCMC_K, b=MCMC_B, device=DEVICE, save_file=save_file
    )
    print("--- Training Finished ---")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--save_file",     type=str,   default="/content/drive/MyDrive/ebm_runs/")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_timesteps",type=int, default=1000)
    p.add_argument("--mcmc_k", type=int, default=30)
    p.add_argument("--mcmc_b", type=float, default=0.1)
    
    args = p.parse_args()
    main(args)