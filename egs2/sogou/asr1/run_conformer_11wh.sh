#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set=train_sogou_fbank_11w_2020Q4_nodup_LID_sort
#train_set=train_sogou_fbank_11w_2020Q4_1th_and_2th_LID_sort
#valid_set=train_sogou_fbank_11w_and_S1_dev_LID_200h
train_set=train_sogou_fbank_11w_2021Q1_bpe5k_nodup_nodev_LID_sort
valid_set=train_sogou_fbank_11w_2021Q1_bpe5k_dev_LID_200h
test_sets="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou testNewLong_sogou test0918_sogou"

asr_config=conf/train_asr_conformer_relPos_swish_16GPU_accgrad1_14E4D_conv15_specAugment_11wh.yaml
inference_config=conf/decode_asr_transformer_lm0.2_5best.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=

./asr_conformer_16GPU_11wh.sh                          \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type extracted                             \
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
