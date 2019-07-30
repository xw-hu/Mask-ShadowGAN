# Mask-ShadowGAN: Learning to Remove Shadows from Unpaired Data


by Xiaowei Hu, Yitong Jiang, Chi-Wing Fu, and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## USR Dataset

Our USR dataset is available for download at [Google Drive](https://drive.google.com/open?id=1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ).


## Citation

@inproceedings{hu2019mask,        
&nbsp;&nbsp;&nbsp;&nbsp;  title={{Mask-ShadowGAN}: Learning to Remove Shadows from Unpaired Data},         
&nbsp;&nbsp;&nbsp;&nbsp;  author={Hu, Xiaowei and Jiang, Yitong and Fu, Chi-Wing and Heng, Pheng-Ann},         
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={ICCV},        
&nbsp;&nbsp;&nbsp;&nbsp;  year={2019},        
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear},       
}

        
## Prerequisites
* Python 3.5
* PyTorch 1.0
* torchvision
* numpy

  
## Train
1. Select the training sets (USR, SRD, or ISTD ) and set the path of the dataset in ```train_Mask-ShadowGAN.py```
2. Run ```train_Mask-ShadowGAN.py```


## Test   
1. Select the testing sets (USR, SRD, or ISTD ) and set the path of the dataset in ```test.py```
2. Run ```test.py```


## Acknowledgments
Code is implemented based on a [clean and readable Pytorch implementation of CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN). We would like to thank Aitor Ruano and the authors of [CycleGAN](https://arxiv.org/abs/1703.10593), Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.


