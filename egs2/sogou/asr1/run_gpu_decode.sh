#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sogou_fbank_500h
dev_set=test8000_sogou
#eval_sets="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testNewLong_sogou testmeeting_cat_agc-1218_sogou"
eval_sets="test8000_sogou testIOS_sogou testDuiHua_sogou"

asr_config=conf/train_asr_conformer_relPos_first_8GPU_accgrad1_500h.yaml
decode_config=conf/decode_asr_transformer_lm0.0.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=

./asr_decode.sh                \
    --audio_format wav                                 \
    --feats_type extracted                             \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --decode_config "${decode_config}"                 \
    --train_set "${train_set}"                         \
    --dev_set "${dev_set}"                             \
    --eval_sets "${eval_sets}"                         \
    --srctexts "data/${train_set}/text" "$@"
