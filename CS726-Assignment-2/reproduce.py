import torch
import numpy as np
import argparse
import dataset
import os
from ddpm import sample, DDPM, NoiseScheduler, train

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
    noise_scheduler = NoiseScheduler(num_timesteps=200, type="sigmoid", beta_start=0.001, beta_end=0.05)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path), map_location=device)
        
    else:
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        data_X, _ = dataset.load_dataset("albatross", device=device)
        data_X = data_X.to(device)

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=128, shuffle=True)
        run_name = f'ddpm_64_200_0.001_0.05_albatross_sigmoid'
        train(model, noise_scheduler, dataloader, optimizer, 40, run_name)
        torch.save(model.state_dict(), args.model_path)

    model.eval()
    samples = sample(model, prior_samples.shape[0], noise_scheduler, deterministic=True)

    np.save(args.output_path, samples.cpu().numpy())
    print(f"Reproduced samples saved to {args.output_path}")

if __name__ == "__main__":
    main()
