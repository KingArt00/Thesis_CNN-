from torchvision import datasets, transforms


def load_dataset(dataset_name):

        num_classes = 10
        in_channels = 1

        train = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))


        test = datasets.MNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))


    
        return (train, test, in_channels, num_classes)
