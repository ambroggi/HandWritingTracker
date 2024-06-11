if __name__ == "__main__":
    import argparse
    import helper_functions as hf
    import model as m
    import model_functions as mf
    import select_dataset
    # import os
    # import process_test_images as pti
    import torch

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "-ct",
        "--confidencethreshold",
        help="good prediction confidence threshold between 0 and 1. Default 0.8",
        required=False,
        default=0.8,
    )
    parser.add_argument(
        "-vt",
        "--variancethreshold",
        help="acceptable confidence variance threshold. Default 0.75",
        required=False,
        default=0.75,
    )
    parser.add_argument(
        "-tr",
        "--trainpath",
        help="path to directory of training dataset of images",
        required=False,
        default="",
    )
    parser.add_argument(
        "-v",
        "--validpath",
        help="path to directory of validation dataset of images",
        required=False,
        default="",
    )
    parser.add_argument(
        "-te",
        "--testpath",
        help="path to directory of test images",
        required=False,
        default="",
    )
    parser.add_argument(
        "-r",
        "--resultpath",
        help="path to directory where results are stored",
        required=False,
        default="",
    )
    parser.add_argument(
        "-c",
        "--buildCSV",
        help="flag to build csv file of results",
        required=False,
        action="store_false",  # default true
    )
    parser.add_argument(
        "-p",
        "--plotResults",
        help="flag to plot images with results",
        required=False,
        action="store_false",  # default true
    )
    parser.add_argument(
        "-o",
        "--overrideExit",
        help="flag to override any early exits",
        required=False,
        action="store_true",  # default false
    )
    # parser.add_argument(
    #     "-u",
    #     "--unknownclasses",
    #     help="list of all index number of unknown classes",
    #     required=False,
    #     default=""
    # )
    parser.add_argument(
        "-d",
        "--dataset",
        help="what dataset to load",
        required=False,
        default="MNIST",
        choices=["MNIST", "Flowers102", "Food101", "FasionMNIST", "Random", "Covertype"]
    )
    args = parser.parse_args()

    # Get / Validate Args
    confThreshold = float(args.confidencethreshold)
    varThreshold = float(args.variancethreshold)
    trainPath = hf.validatePathArgs(args.trainpath, "Training Dataset")
    validPath = hf.validatePathArgs(args.validpath, "Validation Dataset")
    testPath = hf.validatePathArgs(args.testpath, "Testing Images")
    resultPath = hf.validatePathArgs(args.resultpath, "Results")
    dataset = args.dataset
    buildCSV = args.buildCSV
    plotResults = args.plotResults
    overrideExit = args.overrideExit
    unknownclasses = [1, 3, 4]

    if dataset in ["Covertype"]:
        plotResults = False

    if confThreshold > 1 or confThreshold < 0:
        print("Threshold should be between 0 and 1")
        if not overrideExit:
            exit()

    if not (buildCSV or plotResults):
        print(
            "Warning: Results will not be reported or saved. Run again with either -c buildCSV or -p plotResults left as defaulted true."
        )
        if not overrideExit:
            exit()

    # creates needed directories and returns absolute paths
    [
        cwdPath,
        trainPath,
        validPath,
        copyFromTestPath,
        copyToTestPath,
        pccdTestImgPath,
        pccdTestImgClssPath,
        resultPath,
    ] = hf.setMakePaths(trainPath, validPath, testPath, resultPath, dataset)

    batch_size = 32

    get_data = select_dataset.get_data(unknown_classes=unknownclasses, version=dataset)

    # Prepare Training DataSet
    trainset = get_data.get_known(trainPath, train=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True
    )
    classes = trainset.classes

    # Prepare Validation DataSet
    validateset = get_data.get_known(train=False)

    validationloader = torch.utils.data.DataLoader(
        validateset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = get_data.get_unknown()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Create Model
    mod = m.model(dataselection_object=get_data, classes=classes)

    # Train Model
    mf.trainModel(mod, trainloader)

    # Validate Model
    mf.validateModel(mod, validationloader, classes, resultPath)

    # Make Predictions
    mf.predictWithModel(
        testloader, mod, resultPath, buildCSV, plotResults, confThreshold, varThreshold
    )
