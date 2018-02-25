Convolutional Recurrent Neural Network
======================================

This software implements OCR system using CNN + RNN + CTCLoss, inspired by CRNN network.

Usage
-----

`
python ./train.py --help
`

Demo
----

1. Train simple OCR using TestDataset data generator.
Training for ~60-100 epochs.
```
python train.pt --test-init True --test-epoch 10 --output-dir <path_to_folder_with_snapshots>
```

2. Run test for trained model with visualization mode.
```
python test.py --snapshot <path_to_folder_with_snapshots>/crnn_resnet18_10_best --visualize True
```

Dependence
----------
* pytorch 0.3.0 +
* [warp-ctc](https://github.com/SeanNaren/warp-ctc)

Articles
--------

* [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
* [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/citation.cfm?id=1143891)