#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

train_set=train_sogou_raw_500h
valid_set=test8000_sogou_raw
test_sets="test8000_sogou_raw not_on_screen_sogou_raw testIOS_sogou_raw testDuiHua_sogou_raw"

joint_config=conf/enh_asr_tuning/train_asr_tasnet_conformer_500h.yaml
inference_config=conf/decode_asr_transformer.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

./enh_asr_tasnet_conformer_500h.sh                   \
    --lang zh                                        \
    --max_wav_duration 15                            \
    --token_type char                                \
    --use_lm ${use_lm}                               \
    --use_word_lm ${use_wordlm}                      \
    --lm_config "${lm_config}"                       \
    --joint_config "${joint_config}"                   \
    --inference_config "${inference_config}"         \
    --train_set "${train_set}"                       \
    --valid_set "${valid_set}"                       \
    --test_sets "${test_sets}"                       \
    --use_signal_ref false                            \
    --fs "${sample_rate}"                            \
    --ngpu 8                                         \
    --srctexts "data/${train_set}/text" "$@"
