
#CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
#--server server_D --exp exp_jsrt_baseline_hist_alpha3_1 --source-dataset JSRT \
#--seg-loss-function Cldice \
#--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
#--aug-mode True --aug-range aug11 --train-size 0.7
#
#CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
#--server server_D --exp exp_jsrt_baseline_hist_alpha3_2 --source-dataset JSRT \
#--seg-loss-function Cldice \
#--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
#--aug-mode True --aug-range aug11 --train-size 0.7
#
#CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
#--server server_D --exp exp_jsrt_baseline_hist_alpha3_3 --source-dataset JSRT \
#--seg-loss-function Cldice \
#--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
#--aug-mode True --aug-range aug11 --train-size 0.7
#
#CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
#--server server_D --exp exp_jsrt_baseline_hist_alpha3_4 --source-dataset JSRT \
#--seg-loss-function Cldice \
#--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
#--aug-mode True --aug-range aug11 --train-size 0.7
#
#CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py \
#--server server_D --exp exp_jsrt_baseline_hist_alpha3_5 --source-dataset JSRT \
#--seg-loss-function Cldice \
#--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
#--aug-mode True --aug-range aug11 --train-size 0.7

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_jsrt_embed_cos_hist_alpha3_1 --source-dataset JSRT \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss-function cos --embedding-alpha 3 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_jsrt_embed_cos_hist_alpha3_2 --source-dataset JSRT \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss-function cos --embedding-alpha 3 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_jsrt_embed_cos_hist_alpha3_3 --source-dataset JSRT \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss-function cos --embedding-alpha 3 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_jsrt_embed_cos_hist_alpha3_4 --source-dataset JSRT \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss-function cos --embedding-alpha 3 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True

CUDA_VISIBLE_DEVICES=0,1 python3 main_proposed_embedding.py \
--server server_D --exp exp_jsrt_embed_cos_hist_alpha3_5 --source-dataset JSRT \
--seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss-function cos --embedding-alpha 3 \
--optim adam --weight-decay 5e-4 --batch-size 16 --lr 0.001 --lr-schedule 100 120 \
--aug-mode True --aug-range aug11 --train-size 0.7 --arch-ae-detach True
