import torch
import torchvision
import torchvision.transforms as transforms

import helper_functions as hf


class MNIST_data():
    def __init__(self, unknown_classes=[]):
        self.channels = 3

        # Prepare Training Dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.channels),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.unknown_classes = unknown_classes
        self.classcount = 10 - len(unknown_classes)
        self.size_after_convolution = 16 * 4 * 4

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
                    [transforms.Grayscale(num_output_channels=self.channels), transforms.ToTensor()]
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
                [transforms.Grayscale(num_output_channels=self.channels), transforms.ToTensor()]
            ),
        )

        hf.target_remaping(datset, self.unknown_classes)
        hf.filter_class_idx(datset, [x for x in range(len(datset.classes)) if x not in self.unknown_classes])

        return datset



def get_data(version="MNIST", unknown_classes=[]):
    if version == "MNIST":
        return MNIST_data(unknown_classes=unknown_classes)
