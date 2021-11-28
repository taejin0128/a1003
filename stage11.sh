#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=test

./asr.sh \
    --num_nodes 1 \
    --stage 11 \
    --stop_stage 11 \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --speed_perturb_factors "" \
    --ngpu 1 \
    --nj 8 \
    --feats_type extracted \
    --feats_normalize global_mvn \
    --host localhost \
    --use_lm false \
    --token_type char \
    --asr_exp exp/exp01a \
    --asr_config conf/tuning/train_asr_transformer2.yaml \
    --asr_args "--batch_size 32 --accum_grad 4" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_nj 8 \
    --gpu_inference false \
    --inference_asr_model "valid.acc.ave_10best.pth" \
    --srctexts "data/train/text data/dev/text" "$@"
