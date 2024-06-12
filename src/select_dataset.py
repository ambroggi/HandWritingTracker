import torch
import torchvision
import torchvision.transforms.v2 as transforms
# import pandas as pd

import helper_functions as hf
from torch.utils.data import Dataset


class MNIST_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 10 - len(unknown_classes)
        self.size_after_convolution = 16 * 4 * 4
        self.model_scale = 1

    def get_known(self, path: str = False, train=False):
        # If custom training set
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = torchvision.datasets.MNIST(
                root="./src/data",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
                ),
            )

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, self.unknown_classes)

        return datset

    def get_unknown(self):
        # # Preprocess Test Images
        # pti.processTestImageDirectory(
        #     cwdPath, copyFromTestPath, copyToTestPath, pccdTestImgClssPath, overrideExit
        # )

        # # Prepare Testset
        # # If there are no test images, exit early
        # if not os.listdir(pccdTestImgClssPath):
        #     print("No test images available.")
        #     print("Exiting early to avoid crash...")
        #     exit()
        # testset = torchvision.datasets.ImageFolder(pccdTestImgPath, transform=transform)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=2
        # )

        datset = torchvision.datasets.MNIST(
            root="./src/data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            ),
        )

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, [x for x in range(len(datset.classes)) if x not in self.unknown_classes])

        return datset


class Food101_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 101 - len(unknown_classes)
        self.size_after_convolution = 35344
        self.model_scale = 3

    def get_known(self, path: str = False, train=False):
        # If custom training set
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = torchvision.datasets.Food101(
                root="./src/data",
                split="train" if train else "test",
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(size=(200, 200), antialias=True), transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
                )
            )

        datset.targets = torch.tensor(datset._labels)
        datset.data = datset._image_files

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, self.unknown_classes)

        datset._labels = list(datset.targets)
        datset._image_files = datset.data

        return datset

    def get_unknown(self):
        # # Preprocess Test Images
        # pti.processTestImageDirectory(
        #     cwdPath, copyFromTestPath, copyToTestPath, pccdTestImgClssPath, overrideExit
        # )

        # # Prepare Testset
        # # If there are no test images, exit early
        # if not os.listdir(pccdTestImgClssPath):
        #     print("No test images available.")
        #     print("Exiting early to avoid crash...")
        #     exit()
        # testset = torchvision.datasets.ImageFolder(pccdTestImgPath, transform=transform)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=2
        # )

        datset = torchvision.datasets.Food101(
            root="./src/data",
            split="test",
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(size=(200, 200), antialias=True), transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            )
        )

        datset.targets = torch.tensor(datset._labels)
        datset.data = datset._image_files

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, [x for x in range(len(datset.classes)) if x not in self.unknown_classes])

        datset._labels = list(datset.targets)
        datset._image_files = datset.data

        return datset


class Flowers102_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 102 - len(unknown_classes)
        self.classes = {x: x for x in range(102)}
        self.size_after_convolution = 35344
        self.model_scale = 3

    def get_known(self, path: str = False, train=False):
        # If custom training set
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = torchvision.datasets.Flowers102(
                root="./src/data",
                split="train" if train else "test",
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(size=(200, 200), antialias=True), transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
                )
            )

        datset.classes = {x: x for x in range(102)}

        datset.targets = torch.tensor(datset._labels)
        datset.data = datset._image_files

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, self.unknown_classes)

        datset._labels = datset.targets.numpy()
        datset._image_files = datset.data

        return datset

    def get_unknown(self):
        # # Preprocess Test Images
        # pti.processTestImageDirectory(
        #     cwdPath, copyFromTestPath, copyToTestPath, pccdTestImgClssPath, overrideExit
        # )

        # # Prepare Testset
        # # If there are no test images, exit early
        # if not os.listdir(pccdTestImgClssPath):
        #     print("No test images available.")
        #     print("Exiting early to avoid crash...")
        #     exit()
        # testset = torchvision.datasets.ImageFolder(pccdTestImgPath, transform=transform)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=2
        # )

        datset = torchvision.datasets.Flowers102(
            root="./src/data",
            split="test",
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(size=(200, 200), antialias=True), transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            )
        )

        datset.classes = {x: x for x in range(102)}

        datset.targets = torch.tensor(datset._labels)
        datset.data = datset._image_files

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, [x for x in range(len(datset.classes)) if x not in self.unknown_classes])

        datset._labels = datset.targets.numpy()
        datset._image_files = datset.data

        return datset


class FashionMNIST_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 10 - len(unknown_classes)
        self.size_after_convolution = 16 * 4 * 4
        self.model_scale = 1

    def get_known(self, path: str = False, train=False):
        # If custom training set
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = torchvision.datasets.FashionMNIST(
                root="./src/data",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
                ),
            )

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, self.unknown_classes)

        return datset

    def get_unknown(self):

        datset = torchvision.datasets.FashionMNIST(
            root="./src/data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            ),
        )

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, [x for x in range(len(datset.classes)) if x not in self.unknown_classes])

        return datset


class Random_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 10 - len(unknown_classes)
        self.size_after_convolution = 44944
        self.model_scale = 1

    def get_known(self, path: str = False, train=False):
        # If custom training set
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = torchvision.datasets.FakeData(
                num_classes=self.classcount,
                size=10000 if train else 1000,
                transform=transforms.Compose(
                    [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
                ),
            )

        datset.classes = {x: x for x in range(self.classcount)}

        return datset

    def get_unknown(self):

        datset = torchvision.datasets.FakeData(
            num_classes=10 - self.classcount,
            size=1000,
            transform=transforms.Compose(
                [transforms.Grayscale(num_output_channels=self.channels), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            ),
        )

        datset.classes = {x: x for x in range(10 - self.classcount)}

        return datset


class genericData(Dataset):
    def __init__(self, X, y):
        self.data, self.targets = X.clone(), y.clone()
        self.classes = {x: x for x in y.unique()}

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class Covertype():
    def __init__(self, unknown_classes=[]):
        from ucimlrepo import fetch_ucirepo

        # fetch dataset
        covertype = fetch_ucirepo(id=31)

        # assert isinstance(covertype.data.features, pd.DataFrame)

        # pd.DataFrame.to_numpy()

        # data (as pandas dataframes)
        self.X = torch.tensor(covertype.data.features.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(covertype.data.targets.to_numpy()).squeeze(dim=-1).apply_(lambda x: x - 1)

        print(min(self.y))
        print(max(self.y))

        self.classes = self.y.unique()
        self.classcount = len(self.classes) - len(unknown_classes)

        def gener():
            x = 0
            while x < 10000:
                x += 1
                yield True
            while True:
                yield False

        geners = [gener() for x in self.classes]

        undersampling_mask = [next(geners[x]) for x in self.y]

        self.X = self.X[undersampling_mask]
        self.y = self.y[undersampling_mask]

        # metadata
        # print(covertype.metadata)

        # variable information
        # print(covertype.variables)

        train = genericData(self.X, self.y)
        print(self.y.bincount())
        hf.target_remaping(train, unknown_classes)
        print(train.targets.bincount())
        hf.filter_class_idx(train, unknown_classes)
        print(train.targets.bincount())

        self.train, self.val = torch.utils.data.random_split(train, [3 * len(train) // 4, len(train) - (3 * len(train) // 4)])
        self.train.classes = train.classes
        self.val.classes = train.classes

        self.test = genericData(self.X, self.y)
        hf.target_remaping(self.test, unknown_classes)
        hf.filter_class_idx(self.test, [x for x in range(len(self.test.classes)) if x not in unknown_classes])

        self.unknown_classes = unknown_classes
        self.size_after_convolution = 160
        self.model_scale = 4
        self.channels = -1
        self.classcount = len(self.classes) - len(unknown_classes)

    def get_known(self, path: str = False, train=False):
        if path:
            datset = torchvision.datasets.ImageFolder(path, transform=self.transform)
        # else default training set
        else:
            datset = self.train if train else self.val

        return datset

    def get_unknown(self):
        datset = self.test
        return datset


def get_data(version="MNIST", unknown_classes=[]):
    if version == "MNIST":
        return MNIST_data(unknown_classes=unknown_classes)
    elif version == "Food101":
        return Food101_data(unknown_classes=unknown_classes)
    elif version == "Flowers102":
        return Flowers102_data(unknown_classes=unknown_classes)
    elif version == "FasionMNIST":
        return FashionMNIST_data(unknown_classes=unknown_classes)
    elif version == "Random":
        return Random_data(unknown_classes=unknown_classes)
    elif version == "Covertype":
        return Covertype(unknown_classes=unknown_classes)
