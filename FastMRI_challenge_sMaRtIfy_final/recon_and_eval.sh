python reconstruct_final.py \
  --path-hacker-brain-acc4 '../result/PromptMR++_hacker_moe/checkpoints/model_epoch65in65_brain_acc4.pt' \
  --path-hacker-brain-acc8 '../result/PromptMR++_hacker_moe/checkpoints/model_epoch65in65_brain_acc8.pt' \
  --path-hacker-knee-acc4 '../result/PromptMR++_hacker_moe/checkpoints/model_epoch65in65_knee_acc4.pt' \
  --path-hacker-knee-acc8 '../result/PromptMR++_hacker_moe/checkpoints/model_epoch65in65_knee_acc8.pt' \
  --path-helper-brain-acc4 '../result/PromptMR++_helper_moe/checkpoints/model_epoch60in60_brain_acc4.pt' \
  --path-helper-brain-acc8 '../result/PromptMR++_helper_moe/checkpoints/model_epoch60in60_brain_acc8.pt' \
  --path-helper-knee-acc4 '../result/PromptMR++_helper_moe/checkpoints/model_epoch60in60_knee_acc4.pt' \
  --path-helper-knee-acc8 '../result/PromptMR++_helper_moe/checkpoints/model_epoch60in60_knee_acc8.pt'

python leaderboard_eval.py \
  -lp '../Data/leaderboard' \
  -yp '../result/model_final/reconstructions_leaderboard'