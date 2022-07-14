echo inference
python3 inference.py \
--config configs/inference.yaml \
--video_source=docs/demo/videos/RD_Radio34_003_512.mp4 \
--image_source=docs/demo/images/100.jpg \
--cross_id \
--output_dir=docs/demo/output/ \
--frame_limit=100 --inversion_option=encode --if_align --if_extract


python inference.py \
 --config configs/inference.yaml \
 --video_source=./docs/demo/videos/ \
 --output_dir=./docs/demo/output --if_extract