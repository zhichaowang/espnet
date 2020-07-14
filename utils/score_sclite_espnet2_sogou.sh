#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=false
filter=""
num_spkrs=1
help_message="Usage: $0 <data-dir> <dict>"

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1

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


if [ $num_spkrs -eq 1 ]; then
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
#  sed -i "s/<UNK> 2 //g" ${dir}/utt_tra
#  sed -i "s/<UNK> //g" ${dir}/utt_tra

  paste -d" " ${dir}/utt_tra ${dir}/utt_id >${dir}/hyp.trn.tmp
  rm ${dir}/utt_tra ${dir}/utt_id

  sclite -r ${dir}/ref.trn.tmp trn -h ${dir}/hyp.trn.tmp trn -i rm -o all stdout > ${dir}/result.txt
#  sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt
fi
