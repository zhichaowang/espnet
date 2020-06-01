testsets="testCarNoisyDatong_sogou"
flag='utf-8'
for testdata in $testsets; do 
  cp data/$testdata/text data/$testdata/text.gbk
  awk '{print $1}' data/$testdata/text.gbk >data/$testdata/key
  cut -d" " -f2- data/$testdata/text >data/$testdata/tra
  sed -i "s/ //g" data/$testdata/tra
  paste -d" " data/$testdata/key data/$testdata/tra >data/$testdata/text
  rm data/$testdata/tra data/$testdata/key
done
  

