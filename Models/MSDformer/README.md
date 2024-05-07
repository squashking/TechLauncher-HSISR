# Process Overview

## Step 1: Training
**Command:**`python main.py train --dataset_name 'appf' --n_blocks 4 --epochs 200 --batch_size 48 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`

**Explanation:**
- `dataset_name` is the name of the dataset.
- `n_scale` is the magnification factor. Correspondingly, create a folder named `{dataset_name}_x{n_scale}` in the `dataset32` directory. For example, you should now create a folder named "appf_x2" and place `train` and `val` inside it.

## Step 2: Resuming Training
Change the resume flag to `True` on line 28 of the code. Modify the path in `model_name` on line 138 to the path where you need to resume training. Adjust the best values on lines 153 and 154 to ensure the functionality is correct.

## Step 3: Testing
**Command:**`python main.py test --dataset_name 'appf' --n_blocks 4 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`


## Step 4: Obtaining Results
**Command:**`python main.py result --dataset_name 'fin_result' --n_blocks 4 --model_title "MSDformer" --n_subs 8 --n_ovls 2 --n_feats 240 --n_scale 2 --gpus "0,1"`
