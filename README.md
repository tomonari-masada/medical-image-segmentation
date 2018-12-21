## training

### without GPU
```
$ python main.py
```

### with GPU
```
$ python main.py --cuda
```

## prediction

### without GPU
```
$ python predict.py --model model_epoch_200.pth --input_image val/image/000.png --output_filename='temp.png'
```

### with GPU
```
$ python predict.py --model model_epoch_200.pth --input_image val/image/000.png --output_filename='temp.png' --cuda
```
