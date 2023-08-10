# MMSegmentation

Allows processing of images with [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

Uses PyTorch 1.9.0 on CPU.

## Version

MMSegmentation github repo tag/hash:

```
v0.30.0
ed839828760a5f6193822e0bf3492b88ae6140da
```

and timestamp:

```
January 11th, 2023
```

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run --shm-size 8G \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmsegmentation:0.30.0_cpu
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/mmsegmentation:0.30.0_cpu
```

### Build local image

* Build the image from Docker file (from within /path_to/mmsegmentation/0.30.0_cpu)

  ```bash
  docker build -t mmseg .
  ```
  
* Run the container

  ```bash
  docker run --shm-size 8G -v /local/dir:/container/dir -it mmseg
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


### Publish images

#### Build

```bash
docker build -t mmsegmentation:0.30.0_cpu .
```

#### Inhouse registry  

* Tag

  ```bash
  docker tag \
    mmsegmentation:0.30.0_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmsegmentation:0.30.0_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmsegmentation:0.30.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```

#### Docker hub  

* Tag

  ```bash
  docker tag \
    mmsegmentation:0.30.0_cpu \
    waikatodatamining/mmsegmentation:0.30.0_cpu
  ```
  
* Push

  ```bash
  docker push waikatodatamining/mmsegmentation:0.30.0_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login
  ``` 

## Scripts

The following scripts are available:

* `mmseg_config` - for expanding/exporting default configurations (calls [print_config2.py](print_config2.py))
* `mmseg_train` - for training a model (calls `/mmsegmentation/tools/train.py`)
* `mmseg_predict_poll` - for applying a model to images (uses file-polling, calls `/mmsegmentation/tools/predict_poll.py`)
* `mmseg_predict_redis` - for applying a model to images (via [Redis](https://redis.io/) backend), 
  add `--net=host` to the Docker options (calls `/mmsegmentation/tools/predict_redis.py`)
* `mmseg_onnx` - for exporting pytorch models to ONNX (calls `/mmsegmentation/tools/pytorch2onnx.py`)
* `indexed-png-stats` - can output statistics for datasets, i.e., listing the pixel counts per PNG index (for quality checks) 


## Usage

* The annotations must be in indexed PNG format. You can use [wai.annotations](https://github.com/waikato-ufdl/wai-annotations) 
  to convert your data from other formats.
  
* Store class names or label strings in an environment variable called `MMSEG_CLASSES` **(inside the container)**:

  ```bash
  export MMSEG_CLASSES=\'class1\',\'class2\',...
  ```
  
* Alternatively, have the labels stored in a text file with the labels separated by commas and the `MMSEG_CLASSES`
  environment variable point at the file.
  
  * The labels are stored in `/data/labels.txt` either as comma-separated list (`class1,class2,...`) or one per line.
  
  * Export `MMSEG_CLASSES` as follows:

    ```bash
    export MMSEG_CLASSES=/data/labels.txt
    ```

* Use `mmseg_config` to export the config file (of the model you want to train) from `/mmsegmentation/configs` 
  (inside the container), then follow [these instructions](#config).

* Train

  ```bash
  mmseg_train /path_to/your_data_config.py \
      --work-dir /where/to/save/everything
  ```

* Predict and produce PNG files

  ```bash
  mmseg_predict_poll \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --prediction_in /path_to/test_imgs \
      --prediction_out /path_to/test_results
  ```
  Run with `-h` for all available options.

* Predict via Redis backend

  You need to start the docker container with the `--net=host` option if you are using the host's Redis server.

  The following command listens for images coming through on channel `images` and broadcasts
  predicted images on channel `predictions`:

  ```bash
  mmseg_predict_redis \
      --model /path_to/epoch_n.pth \
      --config /path_to/your_data_config.py \
      --redis_in images \
      --redis_out predictions
  ```
  
  Run with `-h` for all available options.


## Example config files

You can output example config files using (stored under `/mmsegmentation/configs` for the various network types):

```bash
mmseg_config \
  --config /mmsegmentation/configs/some/config.py \
  --output_config /output/dir/config.py
```

You can browse the config files [here](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0/configs).


## <a name="config">Preparing the config file</a>

* If necessary, change `num_classes` to number of labels (background not counted).
* Change `dataset_type` to `ExternalDataset` and any occurrences of `type` in the `train`, `test`, `val` 
  sections of the `data` dictionary.
* Change `data_root` to the root path of your dataset (the directory containing `train` and `val` directories).
* In `train_pipeline`, `val_pipeline` and `test_pipeline`: change `img_scale` to preferred values. 
  Image will be scaled to the smaller value between (larger_scale/larger_image_side) and (smaller_scale/smaller_image_side).
* Adapt `img_dir` and `ann_dir` to suit your dataset.
* Interval in `checkpoint_config` will determine the frequency of saving models while training 
  (4000 for example will save a model after every 4000 iterations).
* In the `runner` property, change `max_iters` to how many iterations you want to train the model for.
* Change `load_from` to the file name of the pre-trained network that you downloaded from the model zoo instead
  of downloading it automatically.

_You don't have to copy the config file back, just point at it when training._

**NB:** A fully expanded config file will get placed in the output directory with the same
name as the config plus the extension *.full*.


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) -e USER=$USER ...
```

## Caching models

PyTorch downloads base models, if necessary. However, by using Docker, this means that 
models will get downloaded with each Docker image, using up unnecessary bandwidth and
slowing down the startup. To avoid this, you can map a directory on the host machine
to cache the base models for all processes (usually, there would be only one concurrent
model being trained):  

```
-v /somewhere/local/cache:/.cache
```

Or specifically for PyTorch:

```
-v /somewhere/local/cache/torch:/.cache/torch
```

**NB:** When running the container as root rather than a specific user, the internal directory will have to be
prefixed with `/root`. 


## Testing Redis

You can use [simple-redis-helper](https://pypi.org/project/simple-redis-helper/) to broadcast images 
and listen for image segmentation results when testing.


## Testing inference

You can test the inference of your container with the [image_demo2.py](image_demo2.py) script as follows:

* create a test directory and change into it

  ```bash
  mkdir test_inference
  cd test_inference
  ```

* create cache directory

  ```bash
  mkdir -p cache/torch
  ```

* start the container in interactive mode

  ```bash
  docker run --shm-size 8G -u $(id -u):$(id -g) -e USER=$USER \
    -v `pwd`:/workspace \
    -v `pwd`/cache:/.cache \
    -v `pwd`/cache/torch:/.cache/torch \
    -it public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmsegmentation:0.30.0_cpu 
  ```

* download a pretrained model

  ```bash
  cd /workspace
  mim download mmsegmentation --config pspnet_r50-d8_512x1024_40k_cityscapes --dest .
  ```

* perform inference

  ```bash
  python /mmsegmentation/demo/image_demo2.py \
    --img /mmsegmentation/demo/demo.png \
    --config /mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py \
    --checkpoint pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
    --device cpu \
    --output_file /workspace/demo_out.png
  ```

* the model saved the result of the segmentation in `test_inference/demo_out.png` (in grayscale)  


## Troubleshooting

* `ValueError: SyncBatchNorm expected input tensor to be on GPU`

  Replace `SyncBN` with `BN` in the config file (see [here](https://github.com/open-mmlab/mmsegmentation/issues/387#issuecomment-784712980)).

* Training with only a single class:

  set `num_classes=2` and add parameter `use_sigmoid=False` to the loss function

* Binary segmentation tips: [1](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/faq.md#how-to-handle-binary-segmentation-task), [2](https://github.com/open-mmlab/mmsegmentation/discussions/2757)
