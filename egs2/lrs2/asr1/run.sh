#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=val
test_set="test "

./asr.sh \
    --lang en \
    --train_set ${train_set} \
    --token_type char \
    --lm_config conf/train_lm.yaml \
    --valid_set ${valid_set} \
    --test_sets ${test_set} \
    --asr_config conf/train_asr_transformer.yaml \
    --srctexts "data/${train_set}/text" "$@"
