CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "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" --decoding-strategy "random" --tau 0.5 > out1

CUDA_VISIBLE_DEVICES=0 python task0.py --hf-token "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" --decoding-strategy "random" --tau 0.9 > out2