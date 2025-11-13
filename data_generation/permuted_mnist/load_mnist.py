import os
import sys
import json
import argparse
import torch
import pickle
import torchvision
import torchvision.transforms as transforms


def mnist(arguments):
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    
    
    batch_size = 60000
    #transform = transforms.Compose( [transforms.ToTensor()])
    
    transform = transforms.Compose([ transforms.Resize((7, 7)), transforms.ToTensor() ] )
    
    train_dataset = torchvision.datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data", train=False, transform=transform
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    for i, (images, labels) in enumerate(train_loader):
        images = images.flatten(start_dim=1)
        labels = labels

    x = images
    y = labels

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test

    x_test = images_test
    y_test = labels_test

    #with open(os.path.join(project_root, params['data_dir']), 'wb+') as f:
    #    pickle.dump([x, y, x_test, y_test], f)

    return x, y, x_test, y_test


def get_mnist(type='reg'):
    if type == 'reg':
        data_file = 'data/mnist_'
        with open(data_file, 'rb+') as f:
            x, y, x_test, y_test = pickle.load(f)
    return x, y, x_test, y_test


if __name__ == '__main__':
    """
    Generates all the required data
    """
    
    project_root = os.path.abspath ( os.path.join( os.getcwd(), "..","..") )
    
    confguration_path = os.path.join(project_root,"runtime_config", "data","permuted_mnist","0.json")
    
    sys.exit(mnist( ['-c',  confguration_path  ] ) )# we use the hyperparameters stored in env_temp_cfg to create data for a specifc run




