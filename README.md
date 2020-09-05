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


### Segmentation - Train Examples
* python3 main.py  --exp exp_test --arch unet --train-dataset MC_modified --test-dataset1 JSRT --test-dataset2 SH --batch-size 8 --lr-schedule 100 120 --arg-mode True --arg-thres 0.5 --initial-lr 0.1 --train-size 0.8 


```
python3 main.py  \
--exp exp_test \
--arch unet \
--source-dataset JSRT \
--batch-size 8 \
--lr-schedule 100 120 \
--arg-mode True \
--arg-thres 0.5\
--lr 0.1\
--train-size 0.8
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| work-dir |  [str] 	| Working folder. 	|
| exp 	| [str] 	| ./test/	|
| arch 	|  [str] 	| model architecture. |
| source-dataset 	|  [str] 	| train-dataset. help='JSRT_dataset,MC_dataset,SH_dataset'|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| lr-schedule | [int] 	| number of epochs for training. default : 100 120 |
| lr 	| [float] 	| learning rate. defalut : 0.1	|
| arg-mode | [str] | augmentation mode :  defalut : False|
| arg-thres | [float] | augmentation threshold. default : 0.5|
| train-size| [float] | train dataset size. default : 0.8 |



### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.)






