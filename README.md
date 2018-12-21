# Reference

This implementation is based on [this document (in Japanese)](https://japan-medical-ai.github.io/medical-ai-course-materials/notebooks/Image_Segmentation.html).

# How to use

## Training

### without GPU
```
$ python main.py
```

### with GPU
```
$ python main.py --cuda
```

## Prediction

### without GPU
```
$ python predict.py --model model_epoch_1000.pth --input_image val/image/000.png --output_filename='temp.png'
```

### with GPU
```
$ python predict.py --model model_epoch_1000.pth --input_image val/image/000.png --output_filename='temp.png' --cuda
```
