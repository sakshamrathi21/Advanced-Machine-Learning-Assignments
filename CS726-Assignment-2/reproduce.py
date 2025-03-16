import torch
import numpy as np
import argparse
from ddpm import sample, DDPM, NoiseScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="exps/ddpm_64_200_0.001_0.05_albatross_sigmoid/model.pth")
    parser.add_argument("--prior_samples_path", type=str, default="data/albatross_prior_samples.npy")
    parser.add_argument("--output_path", type=str, default="albatross_samples_reproduce.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_samples = np.load(args.prior_samples_path)
    assert prior_samples.shape[0] == 32561, "Incorrect number of prior samples"
    prior_samples = torch.tensor(prior_samples, dtype=torch.float32, device=device)

    model = DDPM(n_dim=64, n_steps=200).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=200, type="sigmoid", beta_start=0.001, beta_end=0.05)

    samples = sample(model, prior_samples.shape[0], noise_scheduler, deterministic=True)

    np.save(args.output_path, samples.cpu().numpy())
    print(f"Reproduced samples saved to {args.output_path}")

if __name__ == "__main__":
    main()
