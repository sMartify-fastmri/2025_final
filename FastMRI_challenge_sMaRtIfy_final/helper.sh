python train.py \
  -n 'PromptMR++_helper_baseline' \
  -e 30 \
  -f True \
  --enable-hacker False \
  --save-each-epoch True \
  --aug_strength 0

python train_moe.py \
  -n 'PromptMR++_helper_moe' \
  -e 60 \
  -f True \
  --enable-hacker False \
  --save-each-epoch True \
  --starting-expert 'PromptMR++_helper_baseline' \
  --classifier-name 'dummy' \
  --classifier-epoch 1 \
  --classifier-data-path-train '../Data/train/' \
  --classifier-data-path-val '../Data/val/' \
  --starting-expert-max-epoch 30 \
  --starting-epoch 30 \
  --annealing-epoch 15