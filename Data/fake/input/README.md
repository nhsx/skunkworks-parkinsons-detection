# Example slides created by the following invocations from the project root:

## Create PD slides (fake)
```shell
mkdir -p Data/fake/input/PD

python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path Data/fake/input/PD \
  --seed_start 100 \
  --quantity 10
```
## Create Control slides (fake)
```shell
mkdir -p Data/fake/input/Control

python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path Data/fake/input/Control \
  --seed_start 200 \
  --quantity 10 \
  --is_control
```
