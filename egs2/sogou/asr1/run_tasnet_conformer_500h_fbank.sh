#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sogou_fbank_500h
#train_set=train_sogou_raw_500h
valid_set=test8000_sogou
test_sets="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou"
#test_sets="test8000_sogou_raw not_on_screen_sogou_raw testIOS_sogou_raw testDuiHua_sogou_raw test8000_snr0 test8000_snr5 test8000_snr10 test8000_snr15 test8000_snr20"

joint_config=conf/enh_asr_tuning/train_asr_tasnet_L3_N512_H512_conformer_warmLR_500h_fbank.yaml
inference_config=conf/decode_asr_transformer.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

./enh_asr_tasnet_conformer_500h_fbank.sh                           \
    --lang zh                                        \
    --max_wav_duration 15                            \
    --feats_type extracted                           \
    --token_type char                                \
    --use_lm ${use_lm}                               \
    --use_word_lm ${use_wordlm}                      \
    --lm_config "${lm_config}"                       \
    --joint_config "${joint_config}"                 \
    --inference_config "${inference_config}"         \
    --train_set "${train_set}"                       \
    --valid_set "${valid_set}"                       \
    --test_sets "${test_sets}"                       \
    --use_signal_ref false                           \
    --ngpu 8                                         \
    --srctexts "data/${train_set}/text" "$@"
