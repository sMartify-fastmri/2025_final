#!/bin/bash

for ((EPOCH=35; EPOCH>=31; EPOCH--))
do

echo "▶ Reconstructing Hacker Clutch on epoch ${EPOCH}/35"

python reconstruct_moe.py \
  --brain-acc4-epoch2call ${EPOCH} \
  --brain-acc8-epoch2call ${EPOCH} \
  --knee-acc4-epoch2call ${EPOCH} \
  --knee-acc8-epoch2call ${EPOCH}

echo "▶ Evaluating Hacker Clutch on epoch ${EPOCH}/35"

python leaderboard_eval_separate.py \
  -lp '../Data/leaderboard' \
  -yp '../result/test_PromptMR++_hacker_moe_clutch/reconstructions_leaderboard'

done

for ((EPOCH=60; EPOCH>=56; EPOCH--))
do

echo "▶ Reconstructing Hacker on epoch ${EPOCH}/60"

python reconstruct_moe.py \
  --brain-acc4-epoch2call ${EPOCH} \
  --brain-acc8-epoch2call ${EPOCH} \
  --knee-acc4-epoch2call ${EPOCH} \
  --knee-acc8-epoch2call ${EPOCH} \
  --call-moe-name 'test_PromptMR++_hacker_moe' \
  --net_name 'test_PromptMR++_hacker_moe' \
  --max-epoch2call 60

echo "▶ Evaluating Hacker on epoch ${EPOCH}/60"

python leaderboard_eval_separate.py \
  -lp '../Data/leaderboard' \
  -yp '../result/test_PromptMR++_hacker_moe/reconstructions_leaderboard'

done