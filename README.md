# Vilbert-VSNLI

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/devdut1999/Vilbert-VSNLI.git
cd vilbert-multi-task
conda install opencv
pip install -r requirements.txt
```

2. Install pytorch
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install yacs
```
3. Update GCC version to 4.9 and above
```
conda install -c conda-forge gcc
conda install -c conda-forge gxx
```

4. Adding directory path to python
```
export PYTHONPATH="${PYTHONPATH}:/nas/home/devadutt/Vilbert-VSNLI"
```

5. Running the bash script files 
```
bash download_model.sh
bash download_vsnli_data.sh
bash env_set_up.sh
bash extract_image_features.sh
bash train_vilbert.sh
```

