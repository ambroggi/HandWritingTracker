import os
import glob
import pandas as pd
from sklearn.metrics import f1_score, roc_curve
import torch  # NOTE: Apparently loading Torch and then sklearn causes a warning to be thrown up. The other way around does not.


def get_latest_folder(n_th_latest=None) -> str:
    if n_th_latest is None:
        # https://stackoverflow.com/a/32093754
        return max(glob.glob(os.path.join("runs", '*/')), key=os.path.getmtime)
    else:
        # https://stackoverflow.com/a/32093754
        return sorted(glob.glob(os.path.join("runs", '*/')), key=os.path.getmtime)[-n_th_latest-1]


def get_latest_file(n_th_latest=None) -> str:
    latest_folder = get_latest_folder(n_th_latest)
    pth = os.path.join(latest_folder, "results", "logits.csv")
    return pth


def pandas_importing(path: str | os.PathLike = None) -> pd.DataFrame:
    if path is None:
        path = get_latest_file()

    with open(path) as f:
        number_of_logits = f.readline().count(",")
    column_names = [x for x in range(number_of_logits)] + ["True Class"]
    csv = pd.read_csv(path, header=None, names=column_names)
    # print(csv.head())
    return csv


_roc_values_functions = {"Soft": lambda x: x.softmax(dim=1).max(dim=1)[0], "Var": lambda x: x.var(dim=1)}


def get_SoftMax_F1(csv: pd.DataFrame = None, threshold=0.5) -> float:
    return get_F1(version="Soft", csv=csv, threshold=threshold)


def get_VarMax_F1(csv: pd.DataFrame = None, threshold=1) -> float:
    return get_F1(version="Var", csv=csv, threshold=threshold)


def get_F1(version: str, csv: pd.DataFrame = None, threshold=1) -> float:
    if csv is None:
        csv = pandas_importing()
    tense = torch.tensor(csv.iloc[:, :-1].to_numpy())
    prediction = tense.argmax(dim=1)
    unknowns = _roc_values_functions[version](tense).less(threshold)
    prediction[unknowns] = -1
    f1 = f1_score(csv.iloc[:, -1], prediction.numpy(), average="weighted")
    print(f"{version}max F1: {f1}")
    return f1


def get_both_F1(csv: pd.DataFrame = None, threshold_soft=0.5, threshold_var=1) -> tuple[float, float]:
    if csv is None:
        csv = pandas_importing()
    soft_f1 = get_SoftMax_F1(csv, threshold_soft)
    var_f1 = get_VarMax_F1(csv, threshold_var)
    return soft_f1, var_f1


def get_average_F1(path: str | os.PathLike = None, n_samples: int = 3) -> tuple[float, float]:
    f1s = pd.DataFrame([], columns=["Soft", "Var"])
    for x in range(n_samples):
        try:
            pth = get_latest_file(n_th_latest=x)
            f1s.loc[x] = get_both_F1(pandas_importing(pth))
        except FileNotFoundError as e:
            e
            print("Could not find all files.")
            break

    print(f1s)
    soft_mean = f1s["Soft"].mean()
    var_mean = f1s["Var"].mean()
    print(f"Soft Mean: {soft_mean:0.3f}, Var Mean: {var_mean:0.3f}")
    return soft_mean, var_mean


def get_ROC_soft(src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    return get_ROC(version="Soft", src_pth=src_pth, dst_pth=dst_pth)


def get_ROC_var(src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    return get_ROC(version="Var", src_pth=src_pth, dst_pth=dst_pth)


def get_ROC(version: str, src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    if src_pth is None:
        src_pth = get_latest_folder()
    if dst_pth is None:
        dst_pth = src_pth
    if ".csv" not in src_pth:
        csv = pandas_importing(os.path.join(src_pth, "results", "logits.csv"))
    else:
        csv = pandas_importing(src_pth)

    tense = torch.tensor(csv.iloc[:, :-1].to_numpy())
    mx = _roc_values_functions[version](tense)
    labels = csv.iloc[:, -1] == -1
    roc = pd.DataFrame(roc_curve(labels, mx, pos_label=True))
    roc.to_csv(os.path.join(dst_pth, f"{version}maxROC.csv"))
    return roc


if __name__ == "__main__":
    print(get_average_F1())
    get_ROC_soft()
    get_ROC_var()
    print("Done!")
