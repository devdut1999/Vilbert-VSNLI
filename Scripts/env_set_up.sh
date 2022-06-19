cd ..

python3 setup.py develop

cd tools
cd refer
python3 setup.py build_ext --inplace
cd ..
cd ..

git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark
cd vqa-maskrcnn-benchmark
python setup.py build develop
cd ..

cd Scripts
