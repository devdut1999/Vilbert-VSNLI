# Vilbert-VSNLI

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
git clone --recursive https://github.com/devdut1999/Vilbert-VSNLI.git
cd Vilbert-VSNLI
conda env create --file=environment.yml
conda activate dev
```


2. Adding directory path to python
```
export PYTHONPATH="${PYTHONPATH}:/nas/home/devadutt/Vilbert-VSNLI"
```

3. Running the bash script files 
```
cd Scripts
bash download_model.sh          - download pretrained models
bash download_vsnli_data.sh     - download vsnli data
bash env_set_up.sh              - setup environment
bash extract_image_features.sh  - Extract features from images
bash train_vilbert.sh           - Training the vilbert model
```

