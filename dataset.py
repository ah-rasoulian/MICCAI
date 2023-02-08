import re
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import random
from collections import Counter


def find_sub_ses_pairs(data_path: str):
    all_sub_ses = []
    for folders in os.listdir(data_path):
        sub = re.findall(r"sub-\d+", folders)[0]
        ses = re.findall(r"ses-\w{6}\d+", folders)[0]
        sub_ses = "{}_{}".format(sub, ses)
        all_sub_ses.append(sub_ses)

    return all_sub_ses


def cross_validation_division(num_folds, out_dir):
    data_path = r'C:\lausanne-aneurysym-patches\Data_Set_Feb_05_2023_v1-all'
    patients = find_sub_ses_pairs(os.path.join(data_path, "Positive_Patches"))
    patients = list(dict.fromkeys(patients))  # removing duplicates
    all_sub_ses = find_sub_ses_pairs(os.path.join(data_path, "Negative_Patches"))
    controls = [x for x in all_sub_ses if x not in patients]
    assert len(all_sub_ses) == len(patients) + len(controls), "the length of all sub-sessions is not equal to controls and patients"

    y = np.zeros(len(patients) + len(controls))
    y[:len(patients)] = 1
    x = np.concatenate([np.array(patients), np.array(controls)])

    sub_ses_test_folds = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        sub_ses_test_folds.append(x[test_index])

    for i in range(num_folds):
        with open(os.path.join(out_dir, f'fold{i + 1}-test-sub-ses.pth'), 'wb') as file:
            pickle.dump(sub_ses_test_folds[i], file)
