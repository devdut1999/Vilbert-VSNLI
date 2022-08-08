cd ..

python3 setup.py develop

cd tools
cd refer
python3 setup.py build_ext --inplace
cd ..
cd ..

git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git/
mv vqa-maskrcnn-benchmark vqa_maskrcnn_benchmark
cd vqa_maskrcnn_benchmark
python setup.py build develop
cd ..

cd Scripts
