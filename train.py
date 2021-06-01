import os
import inspect

import torch as t
import torchvision.datasets as datasets
import torch
import torch.utils.tensorboard as tb
import torchvision
from model import CNNClassifier, save_model, load_model
from utils import  ConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = CNNClassifier().to(device)


def train(args):
    from os import path
    model = CNNClassifier()
    loss = torch.nn.CrossEntropyLoss()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_cnn5'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_cnn5'), flush_secs=1)

    global_step_train = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)


    ### MNSIT
    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    mnist_testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    if not os.path.exists('cnn.th'):
        epoch = 100
        model.train()

        for ep in range(epoch):
            train_confusionMatrix = ConfusionMatrix()
            valid_confusionMatrix = ConfusionMatrix()
            for i, data in enumerate(train_loader, 0):
                images, labels = data
                frw = model(images)
                train_confusionMatrix.add(frw.argmax(1), labels)
                train_loss = loss(frw, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step_train += 1
                train_logger.add_scalar("loss", train_loss, global_step=global_step_train)
            print(f'Running epoch={ep} with accuracy on train data = {train_confusionMatrix.global_accuracy}')
            train_logger.add_scalar("accuracy", train_confusionMatrix.global_accuracy, global_step=global_step_train)

            for i, validdata in enumerate(test_loader, 0):
                images, labels = validdata
                valid_confusionMatrix.add(model(images).argmax(1), labels)

            print(f'Running epoch={ep} with accuracy on valid data = {valid_confusionMatrix.global_accuracy}')

            valid_logger.add_scalar("accuracy", valid_confusionMatrix.global_accuracy, global_step=global_step_train)

            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step_train)
            scheduler.step(valid_confusionMatrix.global_accuracy)
        model.eval()
        save_model(model)
    else:
        model = load_model(model)


def check():
    model = CNNClassifier()
    model.load_state_dict(torch.load('/cnn.th'))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].type())
    # print(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(p=0.9), RandomVerticalFlip(p=0.9), ToTensor()])')

    args = parser.parse_args()
    train(args)
