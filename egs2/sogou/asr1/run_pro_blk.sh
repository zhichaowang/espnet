
#eval_sets_list="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou testNewLong_sogou testlaoluo testluozhenyu testnews testZhiBo_SellGoods testliyongle testluyu testZhiBo_Game testzuqiu testotherdev_20200916""
# eval_sets_list="test8000_sogou"
eval_sets_list="test8000_sogou not_on_screen_sogou testIOS_sogou testDuiHua_sogou testmeeting_cat_agc-1218_sogou testreport_cat_agc_1-2m-1218_sogou testreport_cat_agc_2-4m-1218_sogou testreport_cat_agc_4-6m-1218_sogou testNewLong_sogou"
blk=0.25
for eval_sets in $eval_sets_list; do
  #src_scp_path=data/ctc_outfeats_1wh_C64L160L24/${eval_sets}.out.scp
  #dst_scp_path=data/ctc_outfeats_1wh_C64L160L24/${eval_sets}.out.scp.blk${blk}
  #dst_ark_path=data/ctc_outfeats_1wh_C64L160L24/${eval_sets}.out.ark.blk${blk}
  src_scp_path=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_1wh/${eval_sets}.out.scp
  dst_scp_path=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_1wh/${eval_sets}.out.scp${blk}
  dst_ark_path=/search/speech/wangzhichao/espnet/espnet-wzc/egs2/sogou/asr1/exp/ctc_prob_feats_1wh/${eval_sets}.out.ark${blk}
  python /nfsdata/tools/ctc_wfst_tools/ctc_probs_tools/pro_blk.py ${src_scp_path} ${dst_scp_path} ${dst_ark_path} ${blk} > log.txt 2>&1 &
done
