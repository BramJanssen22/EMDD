export CUDA_VISIBLE_DEVICES=3

CKPT_PATH='YOUR TRAINED CKPT'

python -u infer.py --ckpt_dir $CKPT_PATH --bs 64 --infer_input_dir ../data/symp_data_w_control --infer_split test --infer_output_dir ./infer_output




python -u infer.py --ckpt_dir 'lightning_logs/version_0/checkpoints/epoch=0-step=720.ckpt' --bs 64 --infer_input_dir ../data/symp_data_w_control --infer_split test --infer_output_dir ./infer_output