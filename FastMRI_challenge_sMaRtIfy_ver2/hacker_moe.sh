python train_moe.py \
  -n 'test_PromptMR++_hacker_moe' \
  -f True \
  --enable-hacker True \
  --save-each-epoch True \
  --starting-expert 'test_PromptMR++_hacker_baseline' \
  --classifier-name 'dummy' \
  --classifier-epoch 1 \
  --classifier-data-path-train '../Data/train/' \
  --classifier-data-path-val '../Data/val/'