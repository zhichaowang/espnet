#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k
nj=16


train_set=train_si284
train_aux_set="wsj_train_si284"
valid_set=cv_dev93
test_sets="test_eval92"

./enh_asr.sh \
    --lang "en" \
    --max_wav_duration 2000 \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/tuning/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --train_aux_set "${train_aux_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_signal_ref false \
    --fs "${sample_rate}" \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --nj ${nj}" \
    --srctexts "data/wsj/train_si284/text data/local/other_text/text" "$@"
