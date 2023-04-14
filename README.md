# CS776-DLCV-Project-IITK

Run commands

Training (Default): 

```shell
cd yolov7
```

```python
python train.py --data 'data/custom.yaml' --weights 'weights/yolov7.pt'
```

To train on the yolov7-tiny architecture [Leaky ReLu] :

```python
 python train.py --weights weights/yolov7-tiny.pt --data data/custom.yaml --cfg cfg/training/yolov7-tiny.yaml --device 0 --epochs 200 --name yolov7-tiny-pipe --hyp data/hyp.scratch.tiny.yaml
```

To train on the yolov7-tiny architecture [Parametric ReLu] :

```python
python train.py --weights weights/yolov7-tiny.pt --data data/custom.yaml --cfg cfg/training/yolov7-tiny-PRelu.yaml --device 1 --epochs 200 --name yolov7-tiny-prelu-pipe --hyp data/hyp.scratch.tiny.yaml
```
To train on the yolov7-nano architecture [Parametric ReLu] :

```python
python train.py --weights weights/yolov7-tiny.pt --data data/custom.yaml --cfg cfg/training/yolov7-nano.yaml --device 0 --epochs 200 --name yolov7-nano-pipe --hyp data/hyp.scratch.tiny.yaml
```
To train on the yolov7 :

```python
python train.py --weights weights/yolov7.pt --data data/custom.yaml --cfg cfg/training/yolov7.yaml --device 1 --epochs 200 --name yolov7-pipe --hyp data/hyp.scratch.p5.yaml
```
To train on the yolov5s

```python
python train.py --weights yolov5s.pt --data data/custom.yaml --name yolov5s-pipe
```

To train on the yolov5n

```python
python train.py --weights yolov5n.pt --data data/custom.yaml --name yolov5n-pipe
```

Pre requisites :

```bash
sudo apt install gfortran python3-scipy

cd yolov7

pip install -r requirements.txt 
```

To resume training (on failure):

```python
python train.py --weights runs/train/exp14/last.pt --resume
```

To detect on video:

```python
python detect.py --weights '../weights/custom-prelu.pt' --img-size 640 --source ../testdata/test2.mp4
```