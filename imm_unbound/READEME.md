python3 train_deltanet.py \
  --data_dir data/mm_T150 \
  --alphabet pm1 \
  --cuda --amp --amp_dtype bf16 \
  --layers 2 --d_model 256 --heads 8 --dropout 0.1 \
  --lr 3e-4 --weight_decay 0.001 \
  --batch_size 256 --num_workers 4 \
  --allow_neg_eigval \
  --save_path ckpt_deltanet_rowtf_step1_clean.pt

python3 train_transformer.py \
  --data_dir data/mm_modm \
  --alphabet pm1 \
  --cuda --amp --amp_dtype bf16 \
  --layers 2 --d_model 256 --heads 8 \
  --ff_mult 4 \
  --lr 3e-4 --weight_decay 0.001 \
  --dropout 0.1 \
  --batch_size 256 --num_workers 4 \
  --early_stop loss \
  --save_path ckpt_transformer_norowtf.pt


python3 train_rnn_relu.py \
  --data_dir data/mm_modm \
  --alphabet pm1 --m_max 97 \
  --cuda --amp --amp_dtype bf16 \
  --layers 2 --d_model 256 --dropout 0.1 \
  --lr 3e-4 --weight_decay 0.001 \
  --batch_size 256 --num_workers 4 \
  --save_path ckpt_rnnrelu_rowtf_step1_clean.pt
