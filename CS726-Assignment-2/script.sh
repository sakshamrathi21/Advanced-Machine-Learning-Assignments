#!/bin/bash

# Define datasets
datasets=("albatross")

# Define T values
T_values=(200)

# Define hyperparameters
lbeta=(0.0001 0.0005 0.001)
ubeta=(0.02 0.05 0.1) 
lr=(0.0001 0.0005 0.001)
n_samples=10000
n_dim=64
batch_size=(32 64 128)
epochs=(30 40 50)

results_file="tuning_results.txt"

# Create results file
echo "Hyperparameter Tuning Results for Albatross (T=200)" > "$results_file"

# Loop over datasets and T values
for dataset in "${datasets[@]}"; do
    for n_steps in "${T_values[@]}"; do
        for lbeta in "${lbeta[@]}"; do
            for ubeta in "${ubeta[@]}"; do
                for lr in "${lr[@]}"; do
                    for batch_size in "${batch_size[@]}"; do
                        for epochs in "${epochs[@]}"; do
                            echo "Training with lbeta=$lbeta, ubeta=$ubeta, lr=$lr, batch_size=$batch_size, epochs=$epochs" | tee -a "$results_file"
                
                            python3 ddpm.py --mode 'train' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" \
                                --epochs "$epochs" --n_samples "$n_samples" --batch_size "$batch_size" \
                                --lr "$lr" --n_dim "$n_dim" --dataset "$dataset" | tee -a "$results_file"

                            python3 ddpm.py --mode 'sample' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" \
                                --epochs "$epochs" --n_samples "$n_samples" --batch_size "$batch_size" \
                                --lr "$lr" --n_dim "$n_dim" --dataset "$dataset" | tee -a "$results_file"

                            echo "---------------------------------------" | tee -a "$results_file"
                        done
                    done
                done
            done
        done
    done
done

echo "Hyperparameter tuning complete. Results saved to $results_file"
