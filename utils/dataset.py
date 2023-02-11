import re
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import random
from collections import Counter
from torch.utils.data import Dataset
import nibabel as nib
import torch


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


def get_train_valid_test_sub_ses(data_path, fold_to_do, folds_dir):
    assert fold_to_do in range(1, 11), "wrong fold number"

    valid_fold = (fold_to_do + 1) % 10
    all_sub_ses = find_sub_ses_pairs(os.path.join(data_path, "Negative_Patches"))
    with open(os.path.join(folds_dir, f'fold{fold_to_do}-test-sub-ses.pth'), 'rb') as file:
        test_sub_ses = pickle.load(file)
    with open(os.path.join(folds_dir, f'fold{valid_fold}-test-sub-ses.pth'), 'rb') as file:
        valid_sub_ses = pickle.load(file)

    train_sub_ses = np.array([x for x in all_sub_ses if x not in test_sub_ses and x not in valid_sub_ses])

    return train_sub_ses, valid_sub_ses, test_sub_ses


class AneurysmDataset(Dataset):
    def __init__(self, root_dir, sub_ses_to_use, transform=None):
        self.images_files = []
        self.labels = []
        self.masks_files = []
        self.transform = transform

        self.read_dataset(root_dir, sub_ses_to_use)
        self.shuffle_dataset()

    def read_dataset(self, root_dir, sub_ses_to_use):
        positive_dir_path = os.path.join(root_dir, "Positive_Patches")
        negative_dir_path = os.path.join(root_dir, "Negative_patches")

        for folder in os.listdir(positive_dir_path):
            sub = re.findall(r"sub-\d+", folder)[0]
            ses = re.findall(r"ses-\w{6}\d+", folder)[0]
            sub_ses = "{}_{}".format(sub, ses)
            if sub_ses in sub_ses_to_use:
                folder_path = os.path.join(positive_dir_path, folder)
                for patch_pair in os.listdir(folder_path):
                    patch_pair_path = os.path.join(folder_path, patch_pair)
                    for patch in os.listdir(patch_pair_path):
                        patch_path = os.path.join(patch_pair_path, patch)
                        self.images_files.append(patch_path)
                        self.labels.append(1)
                        mask_path = patch_path.replace("Positive_Patches", "Positive_Patches_Masks").replace("pos_patch_angio", "mask_patch")
                        self.masks_files.append(mask_path)

        for folder in os.listdir(negative_dir_path):
            sub = re.findall(r"sub-\d+", folder)[0]
            ses = re.findall(r"ses-\w{6}\d+", folder)[0]
            sub_ses = "{}_{}".format(sub, ses)
            if sub_ses in sub_ses_to_use:
                folder_path = os.path.join(negative_dir_path, folder)
                for patch_pair in os.listdir(folder_path):
                    patch_pair_path = os.path.join(folder_path, patch_pair)
                    for patch in os.listdir(patch_pair_path):
                        patch_path = os.path.join(patch_pair_path, patch)
                        self.images_files.append(patch_path)
                        self.labels.append(0)
                        self.masks_files.append(None)

    def shuffle_dataset(self):
        temp = list(zip(self.images_files, self.masks_files, self.labels))
        random.shuffle(temp)
        images_files, masks_files, labels = zip(*temp)
        self.images_files = images_files
        self.masks_files = masks_files
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = torch.FloatTensor(nib.load(self.images_files[item]).get_fdata())
        mask = torch.zeros_like(image) if self.masks_files[item] is None else torch.FloatTensor(nib.load(self.masks_files[item]).get_fdata())
        label = torch.FloatTensor([self.labels[item]])

        if self.transform is not None:
            image = self.transform(image)
        image = image.unsqueeze(0)  # to add a channel -> ch, h, w, d

        return image, mask, label


class CustomBatch:
    def __init__(self, data, only_label: bin):
        transposed_data = list(zip(*data))
        self.input = torch.stack(transposed_data[0], 0)
        if only_label:
            self.target = torch.stack(transposed_data[2], 0)
        else:
            self.target = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.input = self.input.pin_memory()
        self.target = self.target.pin_memory()
        return self


def image_label_collate(batch):
    batch = CustomBatch(batch, True)
    return [batch.input, batch.target]


def image_mask_collate(batch):
    batch = CustomBatch(batch, False)
    return [batch.input, batch.target]
