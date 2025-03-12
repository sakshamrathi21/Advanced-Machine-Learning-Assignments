lbeta=0.0001
ubeta=0.02
n_steps=100
lr=0.001
n_samples=10000
n_dim=2
batch_size=128
epochs=40
dataset=moons
python3 ddpm.py --mode 'train_conditional' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" --epochs "$epochs" --n_samples "$n_samples"  --batch_size "$batch_size" --lr "$lr" --n_dim "$n_dim" --dataset "$dataset"
python3 ddpm.py --mode 'sample_multi_class' --n_steps "$n_steps" --lbeta "$lbeta" --ubeta "$ubeta" --epochs "$epochs" --n_samples "$n_samples" --batch_size "$batch_size" --lr "$lr" --n_dim "$n_dim" --dataset "$dataset"