#!/bin/bash

# Define datasets
datasets=("moons" "circles" "manycircles" "blobs")

# Define T values
T_values=(10 50 100 150 200)

# Define hyperparameters
lbeta=0.0001
ubeta=0.02
n_steps=150
lr=0.0001
n_samples=10000
n_dim=2
batch_size=128
epochs=40

# Loop over datasets and T values
for dataset in "${datasets[@]}"; do
    for n_steps in "${T_values[@]}"; do
        echo "Running for dataset: $dataset with T=$n_steps" >> results
        
        python3 ddpm.py --mode 'train' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" \
                        --epochs "$epochs" --n_samples "$n_samples" --batch_size "$batch_size" \
                        --lr "$lr" --n_dim "$n_dim" --dataset "$dataset"

        python3 ddpm.py --mode 'sample' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" \
                        --epochs "$epochs" --n_samples "$n_samples" --batch_size "$batch_size" \
                        --lr "$lr" --n_dim "$n_dim" --dataset "$dataset" >> results
    done
done
