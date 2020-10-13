
CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
--server server_D --exp exp_baseline_hist_small_1 --source-dataset MC_modified \
--seg-loss-function Cldice \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
--server server_D --exp exp_baseline_hist_small_2 --source-dataset MC_modified \
--seg-loss-function Cldice \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
--server server_D --exp exp_baseline_hist_small_3 --source-dataset MC_modified \
--seg-loss-function Cldice \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
--server server_D --exp exp_baseline_hist_small_4 --source-dataset MC_modified \
--seg-loss-function Cldice \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
--server server_D --exp exp_baseline_hist_small_5 --source-dataset MC_modified \
--seg-loss-function Cldice \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_embed_hist_small_1 --source-dataset MC_modified \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_embed_hist_small_2 --source-dataset MC_modified \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_embed_hist_small_3 --source-dataset MC_modified \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_embed_hist_small_4 --source-dataset MC_modified \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_embed_hist_small_5 --source-dataset MC_modified \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True
