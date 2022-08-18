# CLI Example Walkthrough
A step-by-step demonstrating the fully exercised CLI.

These scripts assume a 'Data' folder structure as encapsulated in docs/datadir.tar

## Create synthetic PD slides (10 examples in repo)
```shell
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path Data/fake/input/PD \
  --seed_start 100 \
  --quantity 10
```
## Create synthetic Control slides (10 examples in repo)
```shell
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path Data/fake/input/Control \
  --seed_start 200 \
  --quantity 10 \
  --is_control
```
## Stain PD slides
```shell
ls Data/fake/input/PD/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full Data/fake/full_stain_dump \
    --output_path_segmented Data/fake/segmented_dump \
    --subdirectory PD
```
## Stain Control slides
```shell
ls Data/fake/input/Control/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full Data/fake/full_stain_dump \
    --output_path_segmented Data/fake/segmented_dump \
    --subdirectory Control
```
## Partition PD into training and validation
```shell
cp Data/fake/segmented_dump/PD/*.png Data/fake/training_dump/train/PD

# Move 25% from training to validation (in this case 223 files)
ls Data/fake/training_dump/train/PD/*.png | \
  shuf -n 223 | \
  xargs -I input_arg -n 1 \
  mv input_arg Data/fake/training_dump/val/PD
```
## Partition Control into training and validation
```shell
cp Data/fake/segmented_dump/Control/*.png Data/fake/training_dump/train/Control

# Move 25% from training to validation (in this case 24 files)
ls Data/fake/training_dump/train/Control/*.png | \
  shuf -n 24 | \
  xargs -I input_arg -n 1 \
  mv input_arg Data/fake/training_dump/val/Control
```
## Train the model
```shell
python polygeist/CLI/train_model.py \
  --training_dump_path Data/fake/training_dump \
  --model_dump_dir Data/fake/model_dump
```
## Validate the model
```shell
python polygeist/CLI/validate_model.py \
  --model_file Data/fake/model_dump/PDNET_checkpoint_X_XX_XX_XX \
  --training_dump_path Data/fake/training_dump \
  --roc_file roc_image.png
```
