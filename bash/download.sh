
mkdir checkpoints
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/Encoder_e4e.pth -O ./checkpoints/Encoder_e4e.pth
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/hfgi.pth -O ./checkpoints/hfgi.pth
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/StyleGAN_e4e.pth -O ./checkpoints/StyleGAN_e4e.pth
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/ffhq_pca.pt -O ./checkpoints/ffhq_pca.pt
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/ffhq_PCA.npz -O ./checkpoints/ffhq_PCA.npz
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/interfacegan_directions-20230323T133213Z-001.zip \
 -O ./checkpoints/interfacegan_directions-20230323T133213Z-001.zip
unzip ./checkpoints/interfacegan_directions-20230323T133213Z-001.zip -d ./checkpoints/
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/stylegan2_d_256.pth -O ./checkpoints/stylegan2_d_256.pth
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/model_ir_se50.pth -O ./checkpoints/model_ir_se50.pth
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/StyleHEAT_visual.pt -O ./checkpoints/StyleHEAT_visual.pt
mkdir ./checkpoints/Deep3D/
wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/epoch_20.pth -O ./checkpoints/epoch_20.pth
mv checkpoints/epoch_20.pth checkpoints/Deep3D/epoch_20.pth
wget https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/BFM_Fitting.zip -O ./checkpoints/BFM_Fitting.zip
unzip ./checkpoints/BFM_Fitting.zip -d ./checkpoints/BFM/
mv ./checkpoints/BFM/BFM_Fitting/* ./checkpoints/BFM/
rm -r ./checkpoints/BFM/BFM_Fitting

wget https://github.com/FeiiYin/StyleHEAT/releases/download/styleheat/videos.zip -O ./checkpoints/videos.zip
unzip ./checkpoints/videos.zip -d ./checkpoints/
rm -rf ./checkpoints/__MACOSX
rm ./checkpoints/videos.zip

rm -rf docs/demo/videos/
rm -rf docs/demo/audios/
mkdir docs/demo/videos/
mkdir docs/demo/audios/
mv ./checkpoints/videos/audios/* docs/demo/audios/
rm -rf ./checkpoints/videos/audios/
mv ./checkpoints/videos/* docs/demo/videos/

# pip install -i https://mirrors.cloud.tencent.com/pypi/simple pydub==0.25.1 yacs==0.1.8 librosa==0.6.0 numba==0.48.0 resampy==0.3.1 imageio-ffmpeg==0.4.7