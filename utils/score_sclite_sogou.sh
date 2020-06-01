#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
num_spkrs=1
help_message="Usage: $0 <data-dir> <dict>"

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1
dic=$2

#function full2half() {
#    sed 'y/ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ/abcdefghihklmnopqrstuvwxyz/' |
#    sed 'y/ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ/abcdefghihklmnopqrstuvwxyz/' |
#    sed 'y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghihklmnopqrstuvwxyz/' |
#    sed 'y/一二三四五六七八九零/1234567890/' |sed 'y/１２３４５６７８９０/1234567890/'
#}
function full2half() {
    sed 'y/ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ/ABCDEFGHIJKLMNOPQRSTUVWXYZ/' |
    sed 'y/ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ/ABCDEFGHIJKLMNOPQRSTUVWXYZ/' |
    sed 'y/abcdefghihklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/' |
    sed 'y/一二三四五六七八九零/1234567890/' |sed 'y/１２３４５６７８９０/1234567890/'
}

concatjson.py ${dir}/data.*.json > ${dir}/data.json

if [ $num_spkrs -eq 1 ]; then
  json2trn.py ${dir}/data.json ${dic} --num-spkrs ${num_spkrs} --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

  if ${remove_blank}; then
      sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
  fi
  if [ -n "${nlsyms}" ]; then
      cp ${dir}/ref.trn ${dir}/ref.trn.org
      cp ${dir}/hyp.trn ${dir}/hyp.trn.org
      filt.py -v ${nlsyms} ${dir}/ref.trn.org > ${dir}/ref.trn
      filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
  fi
  if [ -n "${filter}" ]; then
      sed -i.bak3 -f ${filter} ${dir}/hyp.trn
      sed -i.bak3 -f ${filter} ${dir}/ref.trn
  fi

  export LC_ALL=zh_CN.utf8
  cat ${dir}/ref.trn | awk '{ print $NF}' > ${dir}/utt_id
  #cat text | cut -d" " -f2- | full2half > utt_tra
  cat ${dir}/ref.trn | awk '{{for (i = 1; i < NF; i++) printf("%s ", $i);} printf("\n"); }' | full2half > ${dir}/utt_tra
  paste -d" " ${dir}/utt_tra ${dir}/utt_id >${dir}/ref.trn.tmp

  cat ${dir}/hyp.trn | awk '{ print $NF}' > ${dir}/utt_id
  cat ${dir}/hyp.trn | awk '{{for (i = 1; i < NF; i++) printf("%s ", $i);} printf("\n"); }' | full2half > ${dir}/utt_tra
  sed -i "s/@@ //g" ${dir}/utt_tra
  sed -i "s/CH2EN //g" ${dir}/utt_tra
  sed -i "s/EN2CH //g" ${dir}/utt_tra

  paste -d" " ${dir}/utt_tra ${dir}/utt_id >${dir}/hyp.trn.tmp
  rm ${dir}/utt_tra ${dir}/utt_id

  sclite -r ${dir}/ref.trn.tmp trn -h ${dir}/hyp.trn.tmp trn -i rm -o all stdout > ${dir}/result.txt

  echo "write a CER (or TER) result in ${dir}/result.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.txt

  if ${wer}; then
      if [ -n "$bpe" ]; then
            spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/?V~A/ /g" > ${dir}/ref.wrd.trn
            spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/?V~A/ /g" > ${dir}/hyp.wrd.trn
      else
            sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
            sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
      fi
      sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt

      echo "write a WER result in ${dir}/result.wrd.txt"
      grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
  fi
elif [ ${num_spkrs} -eq 2 ]; then
  ref_trns=""
  hyp_trns=""
  for i in $(seq ${num_spkrs}); do
      ref_trns=${ref_trns}"${dir}/ref${i}.trn "
      hyp_trns=${hyp_trns}"${dir}/hyp${i}.trn "
  done
  json2trn.py ${dir}/data.json ${dic} --num-spkrs ${num_spkrs} --refs ${ref_trns} --hyps ${hyp_trns}

  for n in $(seq ${num_spkrs}); do
      if ${remove_blank}; then
          sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp${n}.trn
      fi
      if [ -n "${nlsyms}" ]; then
          cp ${dir}/ref${n}.trn ${dir}/ref${n}.trn.org
          cp ${dir}/hyp${n}.trn ${dir}/hyp${n}.trn.org
          filt.py -v ${nlsyms} ${dir}/ref${n}.trn.org > ${dir}/ref${n}.trn
          filt.py -v ${nlsyms} ${dir}/hyp${n}.trn.org > ${dir}/hyp${n}.trn
      fi
      if [ -n "${filter}" ]; then
          sed -i.bak3 -f ${filter} ${dir}/hyp${n}.trn
          sed -i.bak3 -f ${filter} ${dir}/ref${n}.trn
      fi
  done

  for (( i=0; i<$((num_spkrs * num_spkrs)); i++ )); do
      ind_r=$((i / num_spkrs + 1))
      ind_h=$((i % num_spkrs + 1))
      sclite -r ${dir}/ref${ind_r}.trn trn -h ${dir}/hyp${ind_h}.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.txt
  done

  echo "write CER (or TER) results in ${dir}/result_r*h*.txt"
  eval_perm_free_error.py --num-spkrs ${num_spkrs} ${dir}/result_r1h1.txt ${dir}/result_r1h2.txt \
      ${dir}/result_r2h1.txt ${dir}/result_r2h2.txt > ${dir}/min_perm_result.json
  sed -n '2,4p' ${dir}/min_perm_result.json

  if ${wer}; then
      for n in $(seq ${num_spkrs}); do
          if [ -n "$bpe" ]; then
              spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref${n}.trn | sed -e "s/?V~A/ /g" > ${dir}/ref${n}.wrd.trn
              spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp${n}.trn | sed -e "s/?V~A/ /g" > ${dir}/hyp${n}.wrd.trn
          else
              sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref${n}.trn > ${dir}/ref${n}.wrd.trn
              sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp${n}.trn > ${dir}/hyp${n}.wrd.trn
          fi
      done
      for (( i=0; i<$((num_spkrs * num_spkrs)); i++ )); do
          ind_r=$((i / num_spkrs + 1))
          ind_h=$((i % num_spkrs + 1))
          sclite -r ${dir}/ref${ind_r}.wrd.trn trn -h ${dir}/hyp${ind_h}.wrd.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.wrd.txt
      done

      echo "write WER results in ${dir}/result_r*h*.wrd.txt"
      eval_perm_free_error.py --num-spkrs ${num_spkrs} ${dir}/result_r1h1.wrd.txt ${dir}/result_r1h2.wrd.txt \
          ${dir}/result_r2h1.wrd.txt ${dir}/result_r2h2.wrd.txt > ${dir}/min_perm_result.wrd.json
      sed -n '2,4p' ${dir}/min_perm_result.wrd.json
  fi
fi
           
