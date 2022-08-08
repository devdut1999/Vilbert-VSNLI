cd ..

#cd datasets
#cd vsnli
#mkdir paco
#cd paco
#mkdir extracted_features
#wget https://storage.googleapis.com/allennlp-public-data/snli-ve/flickr30k_images.tar.gz
#tar zxf flickr30k_images.tar.gz
#cd ..
#cd ..

#sed -i '5,5 s/^/#/' vqa_maskrcnn_benchmark/maskrcnn_benchmark/utils/registry.py
#sed -i "s/torch._six.PY3/torch._six.PY37/" vqa_maskrcnn_benchmark/maskrcnn_benchmark/utils/imports.py
#(head -67 vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/image_list.py ; echo '        return batched_imgs' ; tail -5 vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/image_list.py)  > vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/temp.py
#rm vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/image_list.py
#mv vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/temp.py vqa_maskrcnn_benchmark/maskrcnn_benchmark/structures/image_list.py

python3 script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir "/nas/luka-group/pvli-data/final-datasets/Images" --output_folder "datasets/paco/extracted_features"

cd Scripts
