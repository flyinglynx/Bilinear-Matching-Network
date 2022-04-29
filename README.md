# Bilinear Matching Network

This repository is the official implementation of our CVPR 2022 Paper "Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting". [Link](https://arxiv.org/abs/2203.08354)

In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022
Min Shi, Hao Lu, Chen Feng, Chengxin Liu, Zhiguo Cao<sup>*</sup>

Key Laboratory of Image Processing and Intelligent Control, Ministry of Education
School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, China
<sup>*</sup>  corresponding author.

## Updates
- We are currently organizing a more detailed readme file, with more instructions and discussions on how to build a strong baseline for class-agnostic counting. You can first explore our codes. Feel free to post your questions!
- 23 Apr 2022: Training and inference code is released.

## Installation
Our code has been tested on Python 3.8.5 and PyTorch 1.8.1+cu111. Please follow the official instructions to setup your environment. See other required packages in `requirements.txt`.

## Data Preparation
We train and evaluate our methods on FSC-147 dataset. Please follow the [FSC-147 official repository]() to download and unzip the dataset.  Then, please place the data lists  ``data_list/train.txt``, ``data_list/val.txt`` and ``data_list/test.txt`` in the dataset directory. Final the path structure used in our code will be like :
````
$PATH_TO_DATASET/
├──── gt_density_map_adaptive_384_VarV2
│    ├──── 6146 density maps (.npy files)
│    
├──── images_384_VarV2
│    ├──── 6146 images (.jpg)
│ 
├────annotation_FSC147_384.json (annotation file)
├────ImageClasses_FSC147.txt (category for each image)
├────Train_Test_Val_FSC_147.json (official data splitation file, which is not used in our code)
├────train.txt (We generate the list from official json file)
├────val.txt
├────test.txt

````

## Inference
Please first specify the checkpoint directory and checkpoint files in the configuration `yaml` file. We have included our pretrained model [here](https://www.dropbox.com/s/mr52q8kp9tp7cy9/model_best.pth?dl=0), and the configuration has been set in `./config/test_bmnet+.yaml`. Please place the downloaded `model_best.pth` in `/checkpoint/bmnet+_pretrained`.

Run the following command to conduct inference on the FSC-147 dataset. By default the model will be evaluated on the test set. If you want to test the model on the validation set. Please modify the `list_test` under `DATASET` to `DIR_OF_FSC147_DATASET/val.txt`
````
    cd CODE_DIRECTORY
    python train.py --cfg 'config/test_bmnet+.yaml'
````
Note that, before running inference, you need to modify `DIR_OF_FSC147_DATASET` to the path of  your own FSC-147 dataset. 

* The pretrained checkpoint should produce the same results as reported in the paper, i.e., MAE=15.74, MSE=58.53 on the validation set, MAE=14.62, MSE=91.83 on the test set. 


## Training 
Please first modify `DIR_OF_FSC147_DATASET` and `DIR_FOR_YOUR_CHECKPOINTS` in `config/bmnet+_fsc147.yaml` to train BMNet+, same modifications for training BMNet. 
Run the following command to train BMNet+:

````
    cd CODE_DIRECTORY
    python train.py --cfg 'config/bmnet+_fsc147.yaml'
````
- Training BMNet+ requires less than 12GB memory on a single RTX 3090. The training takes about 1 day.

## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{min2022bmnet,
  title={Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting},
  author={Shi, Min and Hao, Lu and Feng, Chen and Liu, Chengxin and Cao, Zhiguo},
  booktitle={Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognization (CVPR)},
  year={2022}
}

```

## Permission and Disclaimer
This code is only for non-commercial purposes.  Trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.
