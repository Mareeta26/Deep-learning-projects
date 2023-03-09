
Deep learning course

1. Object Detection
Learn a mode to re-train a convolutional neural network adept at object detection. The model should be re-trained to only detect two kinds of objects: cars and trucks.

The pretrained model ( yolov5m.pt) trained on COCO dataset is used and then fine-tuned for detecting 2 objects- cars and trucks.

Instructions to run

>python train_lr.py --batch 20 --weights yolov5m.pt --data dataset.yaml --epochs 30 --img 640 --hyp hyp.finetune.yaml --device 0 --diff-backbone-lr --freeze-backbone
  
>python detect.py --weights runs/train/exp15/weights/best.pt --img 640 --source ../data/images/test/ --save-txt --save-conf

2. Text Classification

Develop predictive models from scratch that can determine, given a recipe, which of 12 categories it falls in, using RNN.
The Recurrent Neural Network(RNN) based model uses stacked bi-directional LSTM (Long Short-Term Memory). The bi-directional LSTMs are used to improve model performance on sequence classification tasks by providing fuller learning on the problem [1].They train two LSTMs on the input sequence instead of one LSTM. In stacked bi-directional LSTMs, the outputs of the forward and backward components of the first layer are passed to the forward and backward components of the second layer respectively.
- Include training and validation text and label files in the same folder.
- Download glove.6B.300d.txt * and provide the location as ‘glove_file’.
* https://nlp.stanford.edu/projects/glove/
