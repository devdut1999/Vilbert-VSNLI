# Vilbert-VSNLI

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/devdut1999/Vilbert-VSNLI.git
cd Vilbert-VSNLI
```

2. Setup Conda Environment
```
```

3. Adding directory path to python
```
export PYTHONPATH="${PYTHONPATH}:/nas/home/devadutt/Vilbert-VSNLI"
```

4. Install apex for distributed training 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

5. Running the bash script files 
```
cd Scripts
bash download_model.sh          - download pretrained models
bash download_vsnli_data.sh     - download vsnli data
bash env_set_up.sh              - setup environment
bash extract_image_features.sh  - Extract features from images
bash train_vilbert.sh           - Training the vilbert model
```

