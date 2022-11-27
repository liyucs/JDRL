# JDRL
The implementation of "Learning Single Image Defocus Deblurring with Misaligned Training Pairs".

# Prerequisites  
- The code has been tested with the following environment
  - Ubuntu 18.04
  - Python 3.7.9
  - PyTorch 1.7.0
  - cudatoolkit 10.0.130
  - NVIDIA TITAN RTX GPU

# Datasets
### Setting A
  - Training dataset: [SDD train set](https://drive.google.com/file/d/1f6WQmBPNp3StdQZVahq9JA5J_5u1h9SN/view?usp=sharing)
  - Testing dataset: [SDD test set](https://drive.google.com/file/d/1f6WQmBPNp3StdQZVahq9JA5J_5u1h9SN/view?usp=sharing)
### Setting B
  - Training dataset: [DPDD train set](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)
  - Testing datasets: [DPDD test set](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) and [RealDOF test set](https://github.com/codeslake/IFAN).

### Preparation (for flow estimation)
Download [pwc-weight](https://drive.google.com/file/d/1ZuZPSx28OMYdKwZlAZ-4KMFzucAZLX_L/view?usp=sharing) and put it under './pwc' folder.

# Test
- Download [mprnet-jdrl-sdd](https://drive.google.com/file/d/1AaVMdxEdcoDH4lFYQFgPz5Hs-E95UnER/view?usp=sharing), [mprnet-jdrl-dpdd](https://drive.google.com/file/d/1LArGY-Gom-0zAEBYvuT3ivJxNbtnaTNy/view?usp=sharing) and [unet-jdrl-sdd](https://drive.google.com/file/d/1o7tI2nj6uLEgCPmx7nh2AvziqIaFTM78/view?usp=sharing), then move them to `./checkpoint`.
```shell
$ cd JDRL
```
- Test on SDD:
```shell
MPRNet*: MPRNet with JDRL trained on SDD:
$ python test.py --test_path './SDD/test/' --checkpoint_path './checkpoint/mprnet-jdrl-sdd.pth' --model 'MPRNet'
UNet*: UNet with JDRL trained on SDD:
$ python test.py --test_path './SDD/test/' --checkpoint_path './checkpoint/unet-jdrl-sdd.pth' --model 'UNet'
```

- Test on DPDD:
```shell
MPRNet*: MPRNet with JDRL trained on DPDD:
$ python test.py --test_path './DPDD/test/' --checkpoint_path './checkpoint/mprnet-jdrl-dpdd.pth' --model 'MPRNet'
```
- Others:

[ifan-jdrl-dpdd](https://drive.google.com/file/d/1bBbpawh3Surf6niaN4oaCH5a5z3ct3H2/view?usp=sharing): IFAN with JDRL trained on DPDD dataset. To test this model, please refer to [IFAN](https://github.com/codeslake/IFAN).

# Train
- SDD: 512*512 image patches have been provided in './SDD/train_patches'.
- DPDD: crop the images of DPDD training set into 512*512 patches using the same settings as [DPDNet](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel).  
After getting the training patches, please organize the training dataset according to our code implementation.
### Start training (on SDD, UNet*)
```shell
$ cd JDRL
$ python train.py
```
To apply JDRL to other models trained on DPDD dataset: (i) initialize the reblurring module: keep the pretrained deblurring model weights fixed, and train the reblurring module for several epochs. (ii) decay the learning rate by 0.1~0.01, and jointly train the reblurring module and deblurring module (your model).
