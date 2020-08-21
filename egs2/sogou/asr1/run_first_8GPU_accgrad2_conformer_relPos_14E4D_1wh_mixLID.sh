#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sogou_fbank_1wh_mix_nodev_nodup_LID
dev_set=train_sogou_fbank_1wh_mix_dev_110h
eval_sets="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou testNewLong_sogou test0918_sogou"
#eval_sets="testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou test0918_sogou"

asr_config=conf/train_asr_conformer_EncDec_relPos_first_8GPU_accgrad2_14E4D_1wh.yaml
decode_config=conf/decode_asr_transformer_lm0.2_5best.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=

./asr_first_8GPU_accgrad2_conformer_relPos_14E4D_1wh_mixLID.sh              \
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
