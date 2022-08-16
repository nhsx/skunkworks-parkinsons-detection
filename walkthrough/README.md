# CLI Example Walkthrough
A step-by-step demonstrating the fully exercised CLI.
## Create PD slides (fake: examples in polygeist/data_faker/example/PD/)
```shell
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path /Data/input_fake/PD \
  --seed_start 100 \
  --quantity 10
```
## Create Control slides (fake: polygeist/data_faker/example/Control/)
```shell
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path /Data/input_fake/Control \
  --seed_start 200 \
  --quantity 10 \
  --is_control
```
## Stain PD slides
```shell
mkdir -p /Data/output_fake/full_stain_dump/PD
mkdir -p /Data/output_fake/segmented_dump/PD

ls /Data/input_fake/PD/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full /Data/output_fake/full_stain_dump \
    --output_path_segmented /Data/output_fake/segmented_dump \
    --subdirectory PD
```
## Stain Control slides
```shell
mkdir -p /Data/output_fake/full_stain_dump/Control
mkdir -p /Data/output_fake/segmented_dump/Control

ls /Data/input_fake/Control/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full /Data/output_fake/full_stain_dump \
    --output_path_segmented /Data/output_fake/segmented_dump \
    --subdirectory Control
```
## Partition PD into training and validation
```shell
mkdir -p /Data/output_fake/training_dump/train/PD
mkdir -p /Data/output_fake/training_dump/val/PD

cp /Data/output_fake/segmented_dump/PD/*.png /Data/output_fake/training_dump/train/PD

# Move 25% from training to validation (in this case 223 files)
ls /Data/output_fake/training_dump/train/PD/*.png | \
  shuf -n 223 | \
  xargs -I input_arg -n 1 \
  mv input_arg /Data/output_fake/training_dump/val/PD
```
## Partition Control into training and validation
```shell
mkdir -p /Data/output_fake/training_dump/train/Control
mkdir -p /Data/output_fake/training_dump/val/Control

cp /Data/output_fake/segmented_dump/Control/*.png /Data/output_fake/training_dump/train/Control

# Move 25% from training to validation (in this case 24 files)
ls /Data/output_fake/training_dump/train/Control/*.png | \
  shuf -n 24 | \
  xargs -I input_arg -n 1 \
  mv input_arg /Data/output_fake/training_dump/val/Control
```
## Train the model
```shell
mkdir -p /Data/output_fake/model_dump

python polygeist/CLI/train_model.py \
  --training_dump_path /Data/output_fake/training_dump \
  --model_dump_dir /Data/output_fake/model_dump
```
## Validate the model
```shell
python polygeist/CLI/validate_model.py \
  --model_file /Data/output_fake/model_dump/PDNET_checkpoint_0_14_48_32 \
  --training_dump_path /Data/output_fake/training_dump \
  --roc_file roc_image.png
```
