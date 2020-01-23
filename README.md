# Exploring VAEs to Empower Self-Supervised Learning
The objective of this research was to utilize unlabelled data to improve image classification for images taken from ImageNet 1000 classes dataset over just using labelled images(which are much lower in number) for training. 

Our approach


## use pretrained vae

NOTE: make sure that `models_vae/best.pt` file exists

create a directory to save training log

`mkdir log`

### (1) fully connected layers

freeze vae weights or don't freeze vae weights

`python train_chopped.py --data-dir data/ssl_data_96 --freeze-vaeweights --batch-size 4000 --log-file log/chopped_freeze --epochs 100`

`python train_chopped.py --data-dir data/ssl_data_96 --batch-size 4000 --log-file log/chopped_no_freeze --epochs 100`

### (2) convolutional layers

freeze vae weights or don't freeze vae weights

`python train_chopped_conv.py --data-dir data/ssl_data_96 --freeze-vaeweights --batch-size 4000 --log-file log/chopped_conv_freeze --epochs 100`

`python train_chopped_conv.py --data-dir data/ssl_data_96 --batch-size 4000 --log-file log/chopped_conv_no_freeze --epochs 100`

NOTE: in case of memory error, decrease the batch size using `--batch-size` argument

NOTE: if the training has stopped after some epochs and you want to train for some more epochs, use the above command lines with the additional argument `--restore-file last.pt`

for example, `python train_chopped_conv.py --data-dir data/ssl_data_96 --freeze-vaeweights --batch-size 4000 --log-file log/chopped_conv_freeze --epochs 100 --restore-file last.pt`
