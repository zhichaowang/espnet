
eval_sets_list="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou testNewLong_sogou testlaoluo testluozhenyu testnews testZhiBo_SellGoods testliyongle testluyu testZhiBo_Game testzuqiu testotherdev_20200916"

model_path=/nfsdata/VideoData_Models/3epoch.pth
config_path=/nfsdata/VideoData_Models/config.yaml
log_file=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_11w_video/tmp.log

for eval_sets in $eval_sets_list; do
  out_ark_path=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_11w_video/${eval_sets}.out.ark
  out_scp_path=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_11w_video/${eval_sets}.out.scp
  python /search/speech/wangzhichao/espnet/espnet-wzc/espnet2/bin/asr_inference_ctc_prob.py --ngpu 0 --data_path_and_name_and_type /search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/dump/extracted/${eval_sets}/feats.scp,speech,kaldi_ark --key_file /search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/dump/extracted/${eval_sets}/feats.scp --asr_train_config ${config_path} --asr_model_file ${model_path} --output_dir ${log_file} --config /search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/conf/decode_asr_transformer_lm0_ctc0.5_5best.yaml  --out_ark_path=${out_ark_path} --out_scp_path=${out_scp_path} > log.txt 2>&1 &
done
