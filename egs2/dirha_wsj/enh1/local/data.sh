#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=3
outdir=${PWD}/DIRHA_WSJ
# Select the reference microphone for the generated databases
# => See ${DIRHA}/Additional_info/Floorplan/*.png for the complete list.
ref_mic=Beam_Circular_Array # Beam_Circular_Array Beam_Linear_Array KA6 L1C
# available at https://github.com/SHINE-FBK/DIRHA_English_wsj/tree/master/Training_IRs
# NOTE: if using the RIRs from the above url, please rename the directories in Training_IRs
#  from {T1_O6,T2_O5,T3_O3} to {T1_06,T2_05,T3_03}, so that our script can recognize them.
IR_folder=/export/b18/xwang/data/ # folders for Impulse responses for WSJ contamination
sph_reader=sph2pipe

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi
if [ ! -e "${DIRHA_WSJ}" ]; then
    log "Fill the value of 'DIRHA_WSJ' of db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"
    # Following the instructions in https://github.com/SHINE-FBK/DIRHA_English_wsj
    #  to generate training and test data (wsj data contaminated with noise and reverberation)
    if ! command -v matlab >/dev/null 2>&1; then
        log "matlab not found."
        exit 1
    fi
    if ! command -v ${sph_reader} >/dev/null 2>&1; then
        log "sph2pipe not found."
        exit 1
    fi

    if [ ! -d "${IR_folder}" ]; then
        url=https://github.com/SHINE-FBK/DIRHA_English_wsj.git
        log "'${IR_folder}' does not exist. Download from '${url}' instead."
        (cd local; git clone ${url})
        # rename the directories so that our script can recognize them
        (
        cd local/DIRHA_English_wsj/Training_IRs;
        mv T1_O6 T1_06;
        mv T2_O5 T2_05;
        mv T3_O3 T3_03
        )
        ln -s "${PWD}/local/DIRHA_English_wsj/Training_IRs" "${IR_folder}"
    else
        num_rirs=$(find "${IR_folder}" -type f -name "*.mat" | wc -l)
        if [[ "$num_rirs" != "207" ]]; then
            log "Error: The number of .mat files in '${IR_folder}' is ${num_rirs} != 207."
            exit 1
        fi
    fi

    cmdfile=$(realpath local/contaminate_wsj.sh)
    echo "#!/bin/bash" > "$cmdfile"

    cat >> "$cmdfile" << EOF
matlab -nodesktop -nodisplay -nosplash -r "addpath('${PWD}/local/tools'); Data_Contamination('$ref_mic','$WSJ1', '$WSJ0', '$DIRHA_WSJ', '$outdir', '$IR_folder', '$sph_reader');exit"
EOF
    chmod +x "$cmdfile"

    # Run Matlab (This takes ~8 hours with ref_mic=Beam_Circular_Array)
    # Expected data directories to be generated (~34 GB in total):
    #   - data/Data/WSJ1_contaminated_mic_${ref_mic}/**/*.wav (96337)
    #   - data/Data/WSJ0_contaminated_mic_${ref_mic}/**/*.wav (35487)
    #   - data/Data/DIRHA_wsj_oracle_VAD_mic_${ref_mic}/{Real,Sim}/**/*.wav (818)
    (
    cd local && \
    log "Log is in local/contaminate_wsj.log" && \
    $train_cmd contaminate_wsj.log "$cmdfile"
    )

    # Validate simulation is successfully finished
    num_wsj1_wavs=$(find "${PWD}/data/Data/WSJ1_contaminated_mic_${ref_mic}" -type f -name "*.wav" | wc -l)
    num_wsj0_wavs=$(find "${PWD}/data/Data/WSJ0_contaminated_mic_${ref_mic}" -type f -name "*.wav" | wc -l)
    num_dirha_wavs=$(find "${PWD}/data/Data/DIRHA_wsj_oracle_VAD_mic_${ref_mic}" -type f -name "*.wav" | wc -l)
    if [[ "${num_wsj1_wavs},${num_wsj0_wavs},${num_dirha_wavs}" != "96337,35487,818" ]]; then
        log "Error: Simulation failed! See local/contaminate_wsj.log for more information"
        exit 1;
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: ASR Data Preparation"

    # train data (data/train_si284_Beam_Circular_Array)
    wsj0_contaminated_folder=WSJ0_contaminated_mic_${ref_mic} # path of the wsj0 training data
    wsj1_contaminated_folder=WSJ1_contaminated_mic_${ref_mic} # path of the wsj1 training data
    local/wsj_data_prep.sh ${outdir}/$wsj0_contaminated_folder/??-{?,??}.? ${outdir}/$wsj1_contaminated_folder/??-{?,??}.? || exit 1;
    local/wsj_format_data.sh ${ref_mic} || exit 1;

    # driha test data (data/dirha_{sim,real}_Beam_Circular_Array)
    DIRHA_wsj_data=${outdir}/DIRHA_wsj_oracle_VAD_mic_${ref_mic}
    local/dirha_data_prep.sh ${DIRHA_wsj_data}/Sim dirha_sim_${ref_mic}  || exit 1;
    local/dirha_data_prep.sh ${DIRHA_wsj_data}/Real dirha_real_${ref_mic}  || exit 1;
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Enhancement Data Preparation"

    # train data (data/train_si284_Beam_Circular_Array)

    # driha test data (data/dirha_sim_Beam_Circular_Array)
    
fi

other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Srctexts Preparation"

    mkdir -p "$(dirname ${other_text})"

    # NOTE(kamo): Give utterance id to each texts.
    zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
	    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
	    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}

    log "Create non linguistic symbols: ${nlsyms}"
    cut -f 2- data/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
