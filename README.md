# Meta-Learned Kernel For Blind Super-Resolution Kernel Estimation

## Tested on

Python3.7, PyTorch 1.7.1

Requirements: lmdb, tqdm, PyYAML, imageio, learn2learn, matplotlib, pyarrow, scipy

## Data Preparation


**Download datasets**

* DIV2K Training set (0001.png-0800.png) and the benchmark sets: Set14, B100, Urban100, DIV2K Validation (0801.png-0900.png). Download links can be found [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets)

**Set path to datasets**

* Edit your `configs/env.yaml` and point argument `data.data_dir` to your data directory. Store all **test** HR images under `{benchmark}/HR` and **train** DIV2K train images in `DIV2K/{data.data_folder}` where argument `data.data_folder` defaults to `'train_HR'`. Your data directory should look something like this if `data.data_dir='/datasets/super_res'`:
```
datasets/super_res$ ls
B100  DIV2K  Set14  Urban100

datasets/super_res/DIV2K$ ls
HR train_HR
```

**Generate benchmark degraded LR images and kernels**

* Edit the bash script `scripts/generate_fkp_benchmarks.sh` pointing variable `datapath` your data directory and prepare the benchmarks provided by [FKP](https://github.com/JingyunLiang/FKP) 

```
cd scripts
chmod +x generate_fkp_benchmarks.sh
vi generate_fkp_benchmarks.sh # edit variable datapath
./generate_fkp_benchmarks.sh
```

## Train

All train experiments will be stored in `runs/{run_name}`. By default, the framework generates a LMDB dataset for both the training HR images as well as their probability maps (one time cost) before training begins for fast data loading.

**Training from scratch**

`python main.py configs/train_metakernelgan.yaml name={run_name} reset=true`

**Resuming a run from the latest model saved**

`python main.py runs/{run_name}/config.yaml load={run_name} resume=true`

## Test

All test experiments will be stored in `runs/{run_name}/{test_run_name}`

**Pretrained model**

Replace `run_name` with `metakernelgan` to load our provided pretrained model.

**Benchmark Results (Average across 5 runs)**

`python main.py runs/{run_name}/config.yaml configs/{test_config} load={run_name} name={test_run_name} train.no_of_tests=5 reset=true`
where `test_config` is
- `test_metakernelgan_benchmark_x2_clean.yaml` for x2 
- `test_metakernelgan_benchmark_x4_clean.yaml` for x4 
- `test_metakernelgan_benchmark_x2_noise_ker.yaml` for x2 with non-Gaussian kernel 
- `test_metakernelgan_benchmark_x2_noise_img.yaml` for x2 with Image noise 

**Custom Images**

`python main.py runs/{run_name}/config.yaml configs/test_metakernelgan_custom.yaml load={run_name} data.custom_data_path="Your custom image folder" train.scale={2 OR 4} name={test_run_name} reset=true`

**Additional example arguments**

- `data.data_test=[\'DIV2K\']` and `data.test_only_images=[\'0812.png\']` to run test on a single image in a benchmark
- `optim.evaluation_task_steps=[0,1,25,200]` to evaluate at adaptation step 0,1,25,200 
- `train.save_kernels=[0,1,25,200]` to save kernels at adaptation step 0,1,25,200 
- `train.save_kernels='range(0,1000,1)'` to save kernels at adaptation step [0..1000)
- `train.save_results=true` to save image at the last specified adaptation step

