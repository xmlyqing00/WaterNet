# WaterNet

This is an official implementation for the paper "WaterNet: An adaptive matching pipeline for segmenting water with volatile appearance"  Computational Visual Media, 2020: 1-14.

The detailed document will be released soon.

## 1 Prepare

### 1.1 Dataset

The [dataset](http://t.lyq.me/?d=waternet_water_v2) includes training data and evaluation data.

### 1.2 Pretrained model

The [link](http://t.lyq.me/?d=waternet_water_pretrained) to download the pretrained model.


## 2 Run

Evaluate the pretrained model

```python
python3 eval_WaterNet.py -c=/path/to/cp_WaterNet_199.pth.tar --model-name=WaterNet -v <video_name>
```

Retrain the model
```python
python3 train_WaterNet.py
```



