cd ..

python3 train_tasks.py --bert_model bert-base-uncased --from_pretrained /nas/home/devadutt/Vilbert-VSNLI/Vsnli-vilbert-transfer-learning/pytorch_model_9.bin  --config_file config/bert_base_6layer_6conect.json --tasks 19 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --num_train_epochs 10 --task_specific_tokens --save_name pvli_transfer

cd Scripts
