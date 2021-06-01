# MNSIT-using-ResNets
Code for the article - Building your own simple ResNets with ~0.99 accuracy on MNSIT


To run the code - 
1. Go into the directory
2. Run - tensorboard --logdir log_directory --reload_interval 1 to launch tensorboard and create a log_directory
3. Run -  python train.py --log_dir=log_directory to run the train file.

You can also add a custom transform to your args as --transform. By default the transforms present are ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(p=0.9), RandomVerticalFlip(p=0.9)
