Step1:训练
训练命令：`python main.py train --dataset_name 'appf' --n_blocks 4 --epochs 200 --batch_size 48 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`

解释：`dataset_name`为数据集名称，`n_scale`为放大的倍数，对应的要在`dataset32`文件夹下，创建名为`{dataset_name}_x{n_scale}`的文件夹，例如现在应该创建“appf_x2"的文件夹，并在里面放入`train`和`val`

Step2:续跑
将代码28行resume标志改为`True`，在138行`model_name`中将路径改为需要续跑的路径。在153和154行将best值改大，以确保功能正常。

Step3:测试
测试命令：`python main.py test --dataset_name 'appf' --n_blocks 4 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`


Step4:结果获取
测试命令：`python main.py result --dataset_name 'fin_result' --n_blocks 4 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`