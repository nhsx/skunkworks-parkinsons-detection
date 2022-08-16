# Example slides created by the following invocations from the project root:

## Create PD slides (fake)
```shell
mkdir polygeist/data_faker/example/PD

python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path polygeist/data_faker/example/PD \
  --seed_start 100 \
  --quantity 10
```
## Create Control slides (fake)
```shell
mkdir polygeist/data_faker/example/Control

python polygeist/CLI/create_fake_slide_dataset.py \
  --output_path polygeist/data_faker/example/Control \
  --seed_start 200 \
  --quantity 10 \
  --is_control
```
