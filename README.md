# StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pretrained StyleGAN (ECCV 2022)

[paper](https://arxiv.org/pdf/2203.04036.pdf) | [project website](https://FeiiYin.github.io/StyleHEAT/)

  
<img src="docs/images/402_poster.jpg" width="800px"/> 

## News: Audio-Reenactment Module Incoporated

We incorporate [SadTalker](https://github.com/Winfredy/SadTalker) into our framework to support audio-driven talking head. Thanks for their awesome work!

We add a script for pre-processing checkpoints in `bash/download.sh`.

 
## Abstract

We investigate the latent feature space of a pre-trained StyleGAN and discover some excellent spatial transformation properties. 
Based on the observation, we propose a novel unified framework based on a pre-trained StyleGAN that enables a set of powerful functionalities, 
*i.e.,* *high-resolution video generation, disentangled control by driving video or audio, and flexible face editing*. 

## Environment

```
git clone https://github.com/FeiiYin/StyleHEAT.git
cd StyleHEAT
conda create -n StyleHEAT python=3.7
conda activate StyleHEAT
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements
```

## Quick Start

#### Pretrained Models

Please directly using `bash bash/download.sh` to pre-process the checkpoints.

Or you can manually download our [pre-trained model](https://drive.google.com/drive/folders/1-m47oPsa3kxjgK5eSJ8g8sHzG4zr2oRc?usp=sharing) and put it in ./checkpoints.

| Model | Description
| :--- | :----------
|checkpoints/Encoder_e4e.pth | Pre-trained E4E StyleGAN Inversion Encoder.
|checkpoints/hfgi.pth | Pre-trained HFGI StyleGAN Inversion Encoder.
|checkpoints/StyleGAN_e4e.pth | Pre-trained StyleGAN.
|checkpoints/ffhq_pca.pt | StyleGAN editing directions.
|checkpoints/ffhq_PCA.npz | StyleGAN optimization parameters.
|checkpoints/interfacegan_directions/ | StyleGAN editing directions.
|checkpoints/stylegan2_d_256.pth | Pre-trained StyleGAN discriminator.
|checkpoints/model_ir_se50.pth | Pre-trained id-loss discriminator.
|checkpoints/StyleHEAT_visual.pt | Pre-trained StyleHEAT model.
|checkpoints/BFM | 3DMM library. (Note the zip file should be unzipped to BFM/.)
|checkpoints/Deep3D/epoch_20.pth | Pre-trained 3DMM extractor.

We also provide some example videos along with their corresponding 3dmm parameters in [videos.zip](https://drive.google.com/drive/folders/1-m47oPsa3kxjgK5eSJ8g8sHzG4zr2oRc?usp=sharing).
Please unzip and put them in `docs/demo/videos/` for later inference.

#### Inference

+ Same-Identity Reenactment with a video.
```
python inference.py \
 --config configs/inference.yaml \
 --video_source=./docs/demo/videos/RD_Radio34_003_512.mp4 \
 --output_dir=./docs/demo/output --if_extract
```

+ Cross-Identity Reenactment with a single image and a video.
```
python inference.py \
 --config configs/inference.yaml \
 --video_source=./docs/demo/videos/RD_Radio34_003_512.mp4 \
 --image_source=./docs/demo/images/100.jpg \
 --cross_id --if_extract \
 --output_dir=./docs/demo/output
```
The `--video_source` and `--image_source` can be specified as either a single file or a folder.

For a better inversion result but taking more time, please specify `--inversion_option=optimize` 
and we will optimize the feature latent of StyleGAN-V2. 
Otherwise we will use HFGI encoder to get the style code and inversion condition with `--inversion_option=encode`. 

If you need align (crop) images during the inference process, please specify `--if_align`. 
Or you can first align the source images following FFHQ dataset. 

If you need to extract the 3dmm parameters of the target video during the inference process, please specify `--if_extract`.
Or you can first extract the 3dmm parameters with the script `TODO.sh` and save the 3dmm in the `{video_source}/3dmm/3dmm_{video_name}.npy`

If you only need to edit the expression without modifying the pose, please specify `--edit_expression_only`.  

+ Intuitive Editing. 
```
python inference.py \
 --config configs/inference.yaml \
 --image_source=./docs/demo/images/40.jpg \
 --inversion_option=optimize \
 --intuitive_edit \
 --output_dir=./docs/demo/output \
 --if_extract
```
The 3dmm parameters of the images can also be pre-extracted or online-extracted with the parameter `--if_extract`.

+ Attribute Editing.
```
python inference.py \
 --config configs/inference.yaml \
 --video_source=./docs/demo/videos/RD_Radio34_003_512.mp4 \
 --image_source=./docs/demo/images/40.jpg \
 --attribute_edit --attribute=young \
 --cross_id \
 --output_dir=./docs/demo/output
```  
The support editable attributes include `young`, `old`, `beard`, `lip`. 
Note to preserve the editing attributes details in W space, the optimized inversion method is banned here. 

<!--
Audio Reenactment
```
TODO
```
-->
+ Audio Reenactment.

Please first install SadTalker in the folder of `third_part` as the format of `third_part/SadTalker`.
Download its pre-trained checkpoints according to their instructions.
Install the additional libraries with `pip install pydub==0.25.1 yacs==0.1.8 librosa==0.6.0 numba==0.48.0 resampy==0.3.1 imageio-ffmpeg==0.4.7`.
Then you can run audio reenactment freely.

```
python inference.py \
 --config configs/inference.yaml \
 --audio_path=./docs/demo/audios/RD_Radio31_000.wav \
 --image_source=./docs/demo/images/100.jpg \
 --cross_id --if_extract \
 --output_dir=./docs/demo/output \
 --inversion_option=optimize
```

#### Training

+ Data preprocessing.

1. To train the VideoWarper, please follow [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)
to download and pre-process the VoxCelebA dataset.

2. To train the whole framework, please follow [HDTF](https://github.com/universome/HDTF)
to download the HDTF dataset and see [HDTF-preprocessing](utils/data_preprocess/README.md) to pre-process the dataset.

3. Please follow [PIRenderer](https://github.com/RenYurui/PIRender) to extract the 3DMM parameters and prepare all the data into lmdb files. 

<!--
Or you can directly download the [HDTF-processed]() to get the processed demo dataset.  
-->

Training include 2 stages.

1. Train VideoWarper
```
bash bash/train_video_warper.sh
```


2. Train Video Calibrator
```
bash bash/train_video_styleheat.sh
```

Note several path hyper-parameter of dataset need to be modified and then run the script.

## 🥂 Related Works
- [SadTalker： Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild (SIGGRAPH Asia 2022)](https://github.com/vinthony/video-retalking)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

## Citation
If you find this work useful for your research, please cite:

``` 
@article{2203.04036,
      author = {Yin, Fei and Zhang, Yong and Cun, Xiaodong and Cao, Mingdeng and Fan, Yanbo and Wang, Xuan and Bai, Qingyan and Wu, Baoyuan and Wang, Jue and Yang, Yujiu},
      title = {StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN}, 
      journal = {arxiv:2203.04036},  
      year = {2022}
}
```

## Acknowledgement
Thanks to 
[StyleGAN-2](https://github.com/NVlabs/stylegan2), 
[PIRenderer](https://github.com/RenYurui/PIRender), 
[HFGI](https://github.com/Tengfei-Wang/HFGI), 
[BaberShop](https://github.com/ZPdesu/Barbershop), 
[GFP-GAN](https://github.com/TencentARC/GFPGAN), 
[Pixel2Style2Pixel](https://github.com/eladrich/pixel2style2pixel),
[SadTalker](https://github.com/Winfredy/SadTalker) 
for sharing their code.
