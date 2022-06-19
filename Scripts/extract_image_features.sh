cd ..

cd datasets
cd vsnli
mkdir extracted_featutes
cd ..
cd ..

python3 script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir "datasets/vsnli/flickr30k_images" --output_folder "datasets/vsnli/extracted_features"

cd Scripts