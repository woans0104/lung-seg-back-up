# lung_segmentation


## Information
### Measures

- DICE Coefficient
- Jaccard Coefficient (IoU)
- Average Contour Distance (ACD)
- Average Surface Distance (ASD)

### Data Description 
- Japanese Society of Radiological Technology (JSRT)
- Montgomery County (MC)
- Shenzhen (SZ)
- 참고 논문
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8089599&tag=1
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6663723&tag=1
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6746911&tag=1

### Logger
- Train Logger       : epoch, loss, IoU, Dice, ACD, ASD
- Test Logger        : epoch, loss, IoU, Dice, ACD, ASD


## Getting Started
### Requirements
- Python3 (3.6.8)
- PyTorch (1.2)
- torchvision (0.4)
- NumPy
- pandas
- matplotlib
- medpy
- AdamP
- opencv

### Segmentation - Train Examples
* python3 main_unet.py  --server server_A --exp exp_test --arch unet --source-dataset MC_modified --optim adam --weight-decay 5e-4 --loss-function bce_logit --batch-size 8  --lr 0.1 --lr-schedule 100 120 --aug-mode True --aug-range aug6 --train-size 0.7 

* python3 main_proposed_embedding.py  --server server_A --exp exp_test --source-dataset MC_modified --seg-loss-function Cldice --ae-loss-function Cldice --embedding-loss mse --embedding-alpha 1 --optim adam --weight-decay 5e-4 --batch-size 8 --lr 0.1 --lr-schedule 100 120 --aug-mode True --aug-range aug6 --train-size 0.7 


| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| server |  [str] 	| Server settings. 	|
| work-dir |  [str] 	| Working folder. 	|
| exp 	| [str] 	| folder name. default : /test/	|
| arch 	|  [str] 	| model architecture. |
| source-dataset 	|  [str] 	| train-dataset. help='JSRT_dataset,MC_dataset,SH_dataset'|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| arch 	|  [str] 	| model architecture. |
| arch-ae-detach 	|  [str] 	| autoencoder detach setting. default : True |
| embedding-alpha 	|  [float] 	| embedding loss weight. default : 1 |
| optim 	|  [str] 	| optimizer. choices=['adam','adamp','sgd']. default : sgd |
| loss-function 	|  [str] 	| loss-function. |
| lr-schedule | [int] 	| number of epochs for training. default : 100 120 |
| lr 	| [float] 	| learning rate. defalut : 0.1	|
| aug-mode | [str] | augmentation mode :  defalut : False |
| aug-range | [float] | augmentation range. default : aug6 |
| train-size| [float] | train dataset size. default : 0.7 |



### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.)






