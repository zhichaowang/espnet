#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set=tr_mix_both_min_8k
valid_set=cv_mix_both_min_8k
test_sets="t_mix_both_min_8k"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --spk_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_beamformer_no_wpe.yaml \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    "$@"
