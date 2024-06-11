from datetime import datetime
import matplotlib.pyplot as plt
import os
import torch

# Helper Functions


def validatePathArgs(argPath, argDescStr):
    if not argPath:
        return ""
    elif os.path.isfile(argPath):
        print(
            "{s0} path is a file. Must be directory. {s1}".format(
                s0=argDescStr, s1=argPath
            )
        )
        print("Using default path.")
        return ""
    elif not os.path.isdir(argPath):
        print("{s0} path is not valid: {s1}".format(s0=argDescStr, s1=argPath))
        print("Using default path.")
        return ""
    else:
        print("{s0} path accepted: {s1}".format(s0=argDescStr, s1=argPath))
        return argPath


def setMakePaths(trainPath, validPath, testPath, resultPathParent, datasetName):
    # cwdPath              = absolute path of this file
    # trainpath            = absolute path of training dataset
    # validpath            = absolute path of validation dataset
    # copyFromTestPath     = absolute path of where orginal test images are copied FROM
    # copyToTestPath       = absolute path of where orginal test images are copied TO in results folder
    # pccdTestImgPath      = absolute path of processed test images
    # pccdTestImgClssPath  = absolute path of class file for processed test images
    # resultPath           = absolute path of directory where the result plots and csv will be stored
    # |-cwdPath (..\HandWritingTracker)
    # |----runs
    # |-------datetime
    # |-----------originalTestImages
    # |-----------processedTestImages
    # |---------------images
    # |-----------results
    # |----src
    # |------data
    # |-----------various datasets
    # |-----------testData
    # |-------------- example test data
    # |-------------- testImages - default location to store test images

    # Current Working Directory
    cwdPath = os.path.dirname(os.path.abspath(__file__))

    # resultPathParent = parent folder for this run
    todaynowstr = getTodayNowStr()
    if not resultPathParent:
        resultPathParent = os.path.dirname(cwdPath)
        resultPathParent = os.path.join(resultPathParent, "runs", datasetName, "")
    resultPathParent = os.path.join(resultPathParent, todaynowstr)

    # subfolders for this run:

    # original test images are copied FROM this folder
    if not testPath:
        copyFromTestPath = os.path.join(cwdPath, "data", "testData", "testImages", "")
    else:
        copyFromTestPath = testPath

    # original test images are copied TO this folder
    copyToTestPath = os.path.join(resultPathParent, "originalTestImages", "")

    # processed test images are stored in class folder "images" in this folder
    pccdTestImgPath = os.path.join(resultPathParent, "processedTestImages", "")
    pccdTestImgClssPath = os.path.join(pccdTestImgPath, "images", "")

    # result data is stored in this folder
    resultPath = os.path.join(resultPathParent, "results", "")

    makeFileStructure(copyToTestPath, pccdTestImgClssPath, resultPath)

    # return absolute paths
    paths = [
        cwdPath,
        trainPath,
        validPath,
        copyFromTestPath,
        copyToTestPath,
        pccdTestImgPath,
        pccdTestImgClssPath,
        resultPath,
    ]
    absPaths = absPath(paths)
    return absPaths


# Recursively makes directories
def makeFileStructure(copyTestPath, pccdTestImgClssPath, resultPath):
    os.makedirs(copyTestPath)
    os.makedirs(pccdTestImgClssPath)
    os.makedirs(resultPath)


# Return absolte paths for list of paths
def absPath(paths):
    absPaths = []
    for path in paths:
        if path:
            path = os.path.abspath(path)
        absPaths.append(path)
    return absPaths


# Analyze logits using softmax and varmax
def analyzeLogits(logits: torch.Tensor, varThreshold):
    variance = torch.var(torch.abs(logits), dim=1)
    varmax_mask = variance < varThreshold
    shape = logits.shape
    unknown = torch.zeros(shape[0], device=logits.device)
    unknown[varmax_mask] = -1
    confidence, classif = torch.max(torch.softmax(logits, dim=-1), 1)
    output = torch.stack([classif, confidence, variance, unknown], dim=-1)
    return output


# Write results to CSV
def writeToCSV(resultPath, results):
    resultsCSV = os.path.join(resultPath, "results.csv")
    with open(resultsCSV, "a") as fd:
        for res in results:
            results_str = "{r0},{r1},{r2},{r3}\n".format(
                r0=res[0].item(), r1=res[1].item(), r2=res[2].item(), r3=res[3].item()
            )
            fd.write(results_str)
    print("Batch Result Data Saved to CSV.")


def storeLogits(logPath, logits_tensor, true_names):
    resultsCSV = os.path.join(logPath, "logits.csv")
    with open(resultsCSV, "a") as fd:
        for logits, name in zip(logits_tensor, true_names):
            # https://stackoverflow.com/a/74059533
            results_str = ", ".join([f"{x}" for x in logits]) + f", {name}\n"
            fd.write(results_str)
    print("Logit Data Saved to CSV.")


# Create results plot by batch
def plotImages(images, results, batch, filepath, classif_convert):
    fig = plt.figure(figsize=(36, 36))
    font = {"family": "normal", "weight": "bold", "size": 22}
    font
    plt.rcParams.update({"font.size": 22})
    for i, image in enumerate(images):
        fig.add_subplot(8, 4, i + 1)
        classif, conf, var, varflag = results[i]
        if varflag == 2:
            var = "{:.2f}".format(var)
            label = "var: {r0}".format(r0=var)
        else:
            conf = "{:.2f}".format(conf)
            label = f"class: {classif_convert[int(classif)]}, conf: {conf}"
        if image.numpy()[0].ndim == 2:
            plt.imshow(image.numpy()[0])
            plt.axis("off")
            plt.title(label)
            plt.subplots_adjust(wspace=2, hspace=2)
            plt.tight_layout()
    filepath = os.path.join(filepath, "batch_{b1}.png".format(b1=batch))
    fig.savefig(filepath)
    plt.close()
    print("Plot of Batch Images Saved.")


def getTodayNowStr():
    return (
        datetime.utcnow()
        .strftime("%Y_%m_%d_%H_%M_%S_%f")
    )


def filter_class_idx(dataset: torch.utils.data.Dataset, classes: list[int] = []):
    if hasattr(dataset, "true_names_mapping"):
        invert_mapping_ = {dataset.true_names_mapping[x]: x for x in dataset.true_names_mapping}
        class_ids = [invert_mapping_[x] for x in classes]
    else:
        class_ids = classes

    idx = torch.tensor([True for _ in dataset.targets])
    # https://stackoverflow.com/a/63975459
    for x in class_ids:
        idx &= (dataset.targets != x)
    dataset.targets = dataset.targets[idx]
    if isinstance(dataset.data, torch.Tensor):
        dataset.data = dataset.data[idx]
    else:
        dataset.data = [datum for datum, keep in zip(dataset.data, idx) if keep]
    if hasattr(dataset, "classes"):
        dataset.classes = [y for x, y in enumerate(dataset.classes) if x not in classes]
        # t1 = {x: dataset.class_to_idx[x] for x in dataset.class_to_idx.keys() if x in dataset.classes}
        # t2 = {x: -1 for x in dataset.class_to_idx.keys() if x not in dataset.classes}
        # t1.update(t2)
        # dataset.class_to_idx = t1


def target_remaping(dataset: torch.utils.data.Dataset, classes: list[int] = []):
    move_to_front = [a for a in range(len(dataset.classes)) if a not in classes]
    move_to_back = classes

    mapping_ = {y: x for x, y in enumerate(move_to_front)}
    mapping_.update({y: x+len(mapping_) for x, y in enumerate(move_to_back)})
    dataset.true_names_mapping = {mapping_[x]: x for x in mapping_.keys()}
    dataset.targets.apply_(lambda x: mapping_[x])


# def _target_remap(dataset: torch.utils.data.Dataset, remapping: dict):

#     for old_name

# https://discuss.pytorch.org/t/dataloader-error-trying-to-resize-storage-that-is-not-resizable/177584/2
def collate_fn(batch):
    return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
    }
