#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sogou_raw_500h
valid_set=test8000_sogou_raw
test_sets="test8000_sogou_raw not_on_screen_sogou_raw testIOS_sogou_raw testDuiHua_sogou_raw"

asr_config=conf/train_asr_conformer_relPos_swish_8GPU_accgrad1_warmupLR_500h_raw.yaml
#asr_config=conf/train_forever.yaml
inference_config=conf/decode_asr_transformer.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=

./asr_conformer_8GPU_500h_raw.sh                       \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --srctexts "data/${train_set}/text" "$@"
