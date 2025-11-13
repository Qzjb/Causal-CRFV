# CRFV fever 2-way 
CUDA_VISIBLE_DEVICES="0" python train_CRFV_fever_2way.py \
--seed 1234 \
--batch_size 16 \
--lr 2e-5 \
--epochs 10 \
--weight_decay 2e-3 \
--evi_num 5 \
--max_seq_length 128 

# CRFV politihop 3-way 
CUDA_VISIBLE_DEVICES="0" python train_CRFV_politihop_3way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 5 \
--weight_decay 3e-3 \
--evi_num 20 \
--max_seq_length 128 

# CRFV politihop 2-way 
CUDA_VISIBLE_DEVICES="0" python train_CRFV_politihop_2way.py \
--seed 1234 \
--batch_size 4 \
--lr 1e-5 \
--epochs 10 \
--weight_decay 5e-4 \
--evi_num 20 \
--max_seq_length 128 
