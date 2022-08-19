#!/bin/bash -e

# Integratation test only (N.B. does not produce meaningful classifications)
# Run from the project root folder, e.g. bash ./walkthrough/integration_test.sh

DATADIR="Data/test"
rm -rf $DATADIR

echo "These scripts assume a 'Data' folder structure as encapsulated in docs/datadir.tar"
tar -xvf docs/datadir.tar -C /tmp
mv /tmp/datadir $DATADIR

echo "Create 2 synthetic PD slides"
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path $DATADIR/input/PD \
  --seed_start 100 \
  --quantity 2

echo "Create 2 synthetic Control slides"
python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path $DATADIR/input/Control \
  --seed_start 200 \
  --quantity 2 \
  --is_control

echo "Stain PD slides"
ls $DATADIR/input/PD/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full $DATADIR/full_stain_dump \
    --output_path_segmented $DATADIR/segmented_dump \
    --subdirectory PD

echo "Stain Control slides"
ls $DATADIR/input/Control/*.png | \
  xargs -I input_arg -n 1 -P 1 \
  python polygeist/CLI/preprocess.py \
    --input_slide input_arg \
    --is_synthetic \
    --output_path_full $DATADIR/full_stain_dump \
    --output_path_segmented $DATADIR/segmented_dump \
    --subdirectory Control

echo "Partition PD into training and validation"
cp $DATADIR/segmented_dump/PD/*.png $DATADIR/training_dump/train/PD

echo "Move 25% from training to validation"
NUMPNGS=$(ls $DATADIR/segmented_dump/PD/*.png | wc -l)
NUMPNGS=$(($NUMPNGS / 4))
ls $DATADIR/training_dump/train/PD/*.png | \
  shuf -n $NUMPNGS | \
  xargs -I input_arg -n 1 \
  mv input_arg $DATADIR/training_dump/val/PD

echo "Partition Control into training and validation"
cp $DATADIR/segmented_dump/Control/*.png $DATADIR/training_dump/train/Control

echo "Move 25% from training to validation"
NUMPNGS=$(ls $DATADIR/segmented_dump/Control/*.png | wc -l)
NUMPNGS=$(($NUMPNGS / 4))
ls $DATADIR/training_dump/train/Control/*.png | \
  shuf -n $NUMPNGS | \
  xargs -I input_arg -n 1 \
  mv input_arg $DATADIR/training_dump/val/Control

echo "Train the model (very low batch_size and num_epochs just for integration test)"
python polygeist/CLI/train_model.py \
  --training_dump_path $DATADIR/training_dump \
  --model_dump_dir $DATADIR/model_dump \
  --batch_size 16 \
  --num_epochs 1

echo "Validate the model (very low batch_size just for integration test)"
CHECKPOINT=$(ls $DATADIR/model_dump/PDNET_checkpoint*)
python polygeist/CLI/validate_model.py \
  --model_file $CHECKPOINT \
  --training_dump_path $DATADIR/training_dump \
  --batch_size 16 \
  --roc_file $DATADIR/roc_image.png
