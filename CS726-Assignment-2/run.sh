lbeta=0.0001
ubeta=0.02
python3 ddpm.py --mode 'train' --n_steps 100 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 10 --n_samples 10000 --batch_size 25 --lr 0.01 --n_dim 2 --dataset moons
python3 ddpm.py --mode 'sample' --n_steps 100 --lbeta "$lbeta" --ubeta "$ubeta" --epochs 5 --n_samples 1000000 --batch_size 25 --lr 0.005 --n_dim 2 --dataset moons