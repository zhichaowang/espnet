. path.sh

cmd=run.pl
nj=300
max_jobs_run=40
data=dump/extracted/train_sogou_fbank_11w_and_S1_nodup_nodev_LID
split_dir=$data/splitlog
out_feats_dir=/nfsdata/features/11w_mix_sort
key_file=$data/feats.scp


#mkdir -p $split_dir
#mkdir -p $out_feats_dir
mkdir $out_feats_dir/log

#split_scps=""
#for n in $(seq $nj); do
#    split_scps+=" $split_dir/feats.${n}.scp"
#done

#utils/split_scp.pl $key_file $split_scps

$cmd --max-jobs-run $max_jobs_run JOB=1:$nj $out_feats_dir/log/copy_feats.JOB.log \
  copy-feats --compress=true scp:$split_dir/feats.JOB.scp ark,scp:$out_feats_dir/train.JOB.ark,$out_feats_dir/train.JOB.scp


#for n in $(seq $start_job $end_job); do 
#    copy-feats --compress=true scp:$split_dir/feats.${n}.scp ark,scp:$out_feats_dir/train.${n}.ark,$out_feats_dir/train.${n}.scp &
#done

