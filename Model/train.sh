python3 main.py --train_batch_size 64 --eval_batch_size 16 --kary 30 --output_vocab_size 30 --model_info base --fp_16 0 \
--info 1108_qg_t2 --mode train
# --info webvid_c4v_qg_layer2_trainonly_new --mode train
# --checkpoint_path /root/autodl-tmp/generateSearch/Model/logs/kary:_30_base_k30_c30_dev_recall_dem:_2_ada:_1_adaeff:_1_adanum:_4_RDrop:_0.1_0.15_0_lre2.0d1.0_epoch=3-recall1=911.000000.ckpt --mode eval