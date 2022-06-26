# Vilbert-VSNLI

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/devdut1999/Vilbert-VSNLI.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install opencv
```
3. Update GCC version to 4.9 and above

4. Running the bash script files 
```
bash download_model.sh
bash download_vsnli_data.sh
bash env_set_up.sh
bash extract_image_features.sh
bash train_vilbert.sh
```
5. Adding directory path to python
```
export PYTHONPATH="${PYTHONPATH}:/nas/home/devadutt/Vilbert-VSNLI"
```
