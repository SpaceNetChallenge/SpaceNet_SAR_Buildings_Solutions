# SatShipAI SN6 Solution

We re using the nvidia-docker version of the script with which we install
as system that uses Cuda 10.2 with CuDNN 7.6.5

During the building of the docker container we download 2.5GB of pretrained 
models and copy them into /root/data folder. 

Training script will override these weights on the same directory 
/root/data/.  The folder that contain model weights for inference are:

* effNet4_weightedbce_border
* dn201_weightedbce_border
* srxt50_32x4_weightedbce_border
* rsnt34_weightedbce_border
* Unet_se_resnext50_32x4d_v1_sar_0
* Unet_se_resnext50_32x4d_v1_sar_1
* Unet_se_resnext50_32x4d_v1_sar_2
* Unet_se_resnext50_32x4d_v1_sar_3
* Unet_inceptionresnetv2_v1_sar_1
*Unet_inceptionresnetv2_v1_sar_0
* Unet_inceptionresnetv2_v1_sar_2
* Unet_inceptionresnetv2_v1_sar_3


