cd ..

cd datasets
cd vsnli
mkdir extracted_featutes
wget https://storage.googleapis.com/allennlp-public-data/snli-ve/flickr30k_images.tar.gz
tar zxf flickr30k_images.tar.gz
cd ..
cd ..

sed -i '5,5 s/^/#/' vqa_maskrcnn_benchmark/maskrcnn_benchmark/utils/registry.py
sed -i "s/torch._six.PY3/torch._six.PY37/" vqa_maskrcnn_benchmark/maskrcnn_benchmark/utils/imports.py
python3 script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir "datasets/vsnli/flickr30k_images" --output_folder "datasets/vsnli/extracted_features"

cd Scripts
