import os
import glob
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, recall_score
import torch  # NOTE: Apparently loading Torch and then sklearn causes a warning to be thrown up. The other way around does not.
import plotly.express as px

VAR_BASE_THRESH = 26
SOFT_BASE_TRHESH = 0.98
ENERGY_BASE_THRESH = -7.5
DATASET = "Covertype"   # Or "*"
# DATASET = "MNIST"
# DATASET = "Food101"
# DATASET = "FasionMNIST"
ENERGY_TEMP = 1


def get_latest_folder(n_th_latest=None, dataset=None) -> str:
    if dataset is None:
        dataset = DATASET
    if n_th_latest is None:
        # https://stackoverflow.com/a/32093754
        return max(glob.glob(os.path.join("runs", f'{dataset}/', '*/')), key=os.path.getmtime)
    else:
        # https://stackoverflow.com/a/32093754
        return sorted(glob.glob(os.path.join("runs", f'{dataset}/', '*/')), key=os.path.getmtime)[-n_th_latest - 1]


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


_roc_values_functions = {"Soft": lambda x: x.softmax(dim=1).max(dim=1)[0], "Softthresh": lambda x: x.softmax(dim=1).max(dim=1)[0], "Var": lambda x: x.var(dim=1), "Energy": lambda x: -(ENERGY_TEMP * torch.logsumexp(x / ENERGY_TEMP, dim=1))}


def get_SoftMax_F1(csv: pd.DataFrame = None, threshold=SOFT_BASE_TRHESH) -> float:
    return get_F1(version="Soft", csv=csv, threshold=0)


def get_SoftMax_thresh_F1(csv: pd.DataFrame = None, threshold=SOFT_BASE_TRHESH) -> float:
    return get_F1(version="Softthresh", csv=csv, threshold=threshold)


def get_VarMax_F1(csv: pd.DataFrame = None, threshold=VAR_BASE_THRESH) -> float:
    return get_F1(version="Var", csv=csv, threshold=threshold)


def get_Energy_F1(csv: pd.DataFrame = None, threshold=VAR_BASE_THRESH) -> float:
    return get_F1(version="Energy", csv=csv, threshold=threshold)


def get_F1(version: str, csv: pd.DataFrame = None, threshold=0.5) -> float:
    if csv is None:
        csv = pandas_importing()
    if version == "Soft":
        threshold = 0
    tense = torch.tensor(csv.iloc[:, :-1].to_numpy())
    prediction = tense.argmax(dim=1)
    if version != "Energy":
        unknowns = _roc_values_functions[version](tense).less(threshold)
    else:
        unknowns = _roc_values_functions[version](tense).greater(threshold)
    prediction[unknowns] = -1
    f1 = f1_score(csv.iloc[:, -1], prediction.numpy(), average="weighted")
    print(f"{version}max F1: {f1}")
    return f1


def get_k_uk_SoftMax_F1(csv: pd.DataFrame = None, threshold=SOFT_BASE_TRHESH) -> float:
    return get_k_uk_F1(version="Soft", csv=csv, threshold=0)


def get_k_uk_SoftMax_thresh_F1(csv: pd.DataFrame = None, threshold=SOFT_BASE_TRHESH) -> float:
    return get_k_uk_F1(version="Softthresh", csv=csv, threshold=threshold)


def get_k_uk_VarMax_F1(csv: pd.DataFrame = None, threshold=VAR_BASE_THRESH) -> float:
    return get_k_uk_F1(version="Var", csv=csv, threshold=threshold)


def get_k_uk_Energy_F1(csv: pd.DataFrame = None, threshold=ENERGY_BASE_THRESH) -> float:
    return get_k_uk_F1(version="Energy", csv=csv, threshold=threshold)


def get_k_uk_F1(version: str, csv: pd.DataFrame = None, threshold=0.5) -> tuple[float, float]:
    if csv is None:
        csv = pandas_importing()
    tense = torch.tensor(csv.iloc[:, :-1].to_numpy())
    prediction = tense.argmax(dim=1)
    if version != "Energy":
        unknowns = _roc_values_functions[version](tense).less(threshold)
    else:
        unknowns = _roc_values_functions[version](tense).greater(threshold)
    prediction[unknowns] = -1

    knowns_mask = csv.iloc[:, -1] != -1
    unknowns_mask = csv.iloc[:, -1] == -1

    knowns_f1 = f1_score(csv.loc[knowns_mask].iloc[:, -1], prediction.numpy()[knowns_mask], average="weighted")
    unknowns_f1 = f1_score(csv.loc[unknowns_mask].iloc[:, -1], prediction.numpy()[unknowns_mask], average="weighted")
    # print(f"{version}max knowns F1: {knowns_f1}, unknowns F1: {unknowns_f1}")
    return knowns_f1, unknowns_f1


def get_all_F1(csv: pd.DataFrame = None, threshold_keys={"Soft": 0, "Softthresh": SOFT_BASE_TRHESH, "Var": VAR_BASE_THRESH, "Energy": ENERGY_BASE_THRESH}, split_knowns_unknowns=False) -> pd.DataFrame:
    if csv is None:
        csv = pandas_importing()
    keys = list(threshold_keys.keys())
    if not split_knowns_unknowns:
        f1s = pd.DataFrame([[get_F1(version=x, csv=csv, threshold=threshold_keys[x]) for x in keys]], columns=keys)
    else:
        temp = [get_k_uk_F1(version=x, csv=csv, threshold=threshold_keys[x]) for x in keys]
        temp2 = zip(*temp, [True, False])
        f1s = pd.DataFrame(temp2, columns=keys + ["known"])
    return f1s


def get_average_F1(path: str | os.PathLike = None, n_samples: int = 3, threshold_keys=None) -> pd.DataFrame:
    if threshold_keys is None:
        threshold_keys = {"Soft": 0, "Softthresh": SOFT_BASE_TRHESH, "Var": VAR_BASE_THRESH, "Energy": ENERGY_BASE_THRESH}
    f1s = pd.DataFrame([], columns=threshold_keys.keys())
    for x in range(n_samples):
        try:
            pth = get_latest_file(n_th_latest=x)
            f1s = pd.concat([f1s, get_all_F1(pandas_importing(pth), threshold_keys=threshold_keys)])
        except FileNotFoundError as e:
            e
            print("Could not find all files.")
            break

    print(f1s)
    soft_mean = f1s["Soft"].mean()
    soft_thresh_mean = f1s["Softthresh"].mean()
    var_mean = f1s["Var"].mean()
    energy_mean = f1s["Energy"].mean()
    print(f"Soft Mean: {soft_mean:0.3f}, Soft Threshold Mean: {soft_thresh_mean:0.3f}, Var Mean: {var_mean:0.3f}, Energy Mean: {energy_mean:0.3f}")
    return f1s


def get_average_k_uk_F1(path: str | os.PathLike = None, n_samples: int = 3, threshold_keys=None) -> pd.DataFrame:
    if threshold_keys is None:
        threshold_keys = {"Soft": 0, "Softthresh": SOFT_BASE_TRHESH, "Var": VAR_BASE_THRESH, "Energy": ENERGY_BASE_THRESH}
    f1s = pd.DataFrame([], columns=list(threshold_keys.keys()) + ["known"])
    for x in range(n_samples):
        try:
            pth = get_latest_file(n_th_latest=x)
            f1s = pd.concat([f1s, get_all_F1(pandas_importing(pth), split_knowns_unknowns=True, threshold_keys=threshold_keys)])
        except FileNotFoundError as e:
            e
            print("Could not find all files.")
            break

    print(f1s)
    soft_mean = f1s["Soft"][f1s["known"]].mean(), f1s["Soft"][f1s["known"] != True].mean()
    soft_thresh_mean = f1s["Softthresh"][f1s["known"]].mean(), f1s["Softthresh"][f1s["known"] != True].mean()
    var_mean = f1s["Var"][f1s["known"]].mean(), f1s["Var"][f1s["known"] != True].mean()
    energy_mean = f1s["Energy"][f1s["known"]].mean(), f1s["Energy"][f1s["known"] != True].mean()
    print(f"Soft Mean: {soft_mean[0]:0.3f}|{soft_mean[1]:0.3f}, Soft thresh Mean: {soft_thresh_mean[0]:0.3f}|{soft_thresh_mean[1]:0.3f}, Var Mean: {var_mean[0]:0.3f}|{var_mean[1]:0.3f}, Energy Mean: {energy_mean[0]:0.3f}|{energy_mean[1]:0.3f}")
    return f1s


def get_ROC_soft(src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    return get_ROC(version="Soft", src_pth=src_pth, dst_pth=dst_pth)


def get_ROC_var(src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    return get_ROC(version="Var", src_pth=src_pth, dst_pth=dst_pth)


def get_ROC_energy(src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> pd.DataFrame:
    return get_ROC(version="Energy", src_pth=src_pth, dst_pth=dst_pth)


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


def graph_ROC(version: str, pth: str | os.PathLike = None):
    if pth is None:
        pth = get_latest_folder()
    if ".csv" not in pth:
        csv = pandas_importing(os.path.join(pth, "results", "logits.csv"))
    else:
        csv = pandas_importing(pth)

    tense = torch.tensor(csv.iloc[:, :-1].to_numpy())
    mx = _roc_values_functions[version](tense)
    recall = []

    # All from: https://plotly.com/python/roc-and-pr-curves/
    fpr, tpr, thresholds = roc_curve(csv.iloc[:, -1], mx, pos_label=True)

    for t in thresholds:
        prediction = tense.argmax(dim=1)
        if version != "Energy":
            unknowns = _roc_values_functions[version](tense).less(t)
        else:
            unknowns = _roc_values_functions[version](tense).greater(t)
        prediction[unknowns] = -1
        recall.append(recall_score(csv.iloc[:, -1], prediction.numpy(), average="weighted"))

    # The histogram of scores compared to true labels
    fig_hist = px.histogram(
        x=mx, color=csv.iloc[:, -1], nbins=50,
        labels=dict(color='True Labels', x='Score'), title=f"Bins for {version}"
    )

    fig_hist.show()

    # Evaluating model performance at various thresholds
    df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        "Recall": recall
    }, index=thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"

    fig_thresh = px.line(
        df, title=f'{version}- TPR and FPR at every threshold',
        width=700, height=500
    )

    # fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    fig_thresh.show()


def find_threshold_a(version: str, src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> float:
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
    # roc = roc_curve(labels, mx, pos_label=True if version != "Energy" else False)
    roc = roc_curve(labels, mx, pos_label=True)

    soft_target = recall_score(csv.iloc[:, -1].to_numpy(), tense.argmax(dim=1), average="weighted")

    selected_threshold = 0
    for tpr, fpr, threshold in zip(*roc):
        if tpr > soft_target * 0.8:
            break
        selected_threshold = threshold

    return selected_threshold


def find_threshold_b(version: str, src_pth: str | os.PathLike = None, dst_pth: str | os.PathLike = None) -> float:
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
    # roc = roc_curve(labels, mx, pos_label=True if version != "Energy" else False)
    roc = roc_curve(labels, mx, pos_label=True)

    selected_threshold = 0
    max_recall = 0
    for tpr, fpr, threshold in zip(*roc):
        prediction = tense.argmax(dim=1)
        if version != "Energy":
            unknowns = _roc_values_functions[version](tense).less(threshold)
        else:
            unknowns = _roc_values_functions[version](tense).greater(threshold)
        prediction[unknowns] = -1
        recall = recall_score(csv.iloc[:, -1], prediction.numpy(), average="weighted")
        if recall > max_recall:
            selected_threshold = threshold
            max_recall = recall

    return selected_threshold


if __name__ == "__main__":
    # get_average_F1()
    # get_average_k_uk_F1()
    # # get_ROC_soft()
    # # get_ROC_var()
    # graph_ROC("Soft")
    # graph_ROC("Var")
    # graph_ROC("Energy")
    # print("Done!")
    threshold_keys = None
    # threshold_keys = {"Soft": 0, "Softthresh": find_threshold_a("Softthresh"), "Var": find_threshold_a("Var"), "Energy": find_threshold_a("Energy")}
    threshold_keys = {"Soft": 0, "Softthresh": find_threshold_b("Softthresh"), "Var": find_threshold_b("Var"), "Energy": find_threshold_b("Energy")}
    get_average_F1(threshold_keys=threshold_keys)
    get_average_k_uk_F1(threshold_keys=threshold_keys)
