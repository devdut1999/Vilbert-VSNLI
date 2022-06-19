cd ..

mkdir models
cd models
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin
cd ..

mkdir data
cd data 
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
cd ..

cd Scripts
