import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import random
from collections import Counter
from torch.utils.data import Dataset
import nibabel as nib
import torch
from monai.transforms import NormalizeIntensity


def find_sub_ses_pairs(data_path: str):
    all_sub_ses = []
    for folders in os.listdir(data_path):
        sub = re.findall(r"sub-\d+", folders)[0]
        ses = re.findall(r"ses-\w{6}\d+", folders)[0]
        sub_ses = "{}_{}".format(sub, ses)
        all_sub_ses.append(sub_ses)

    return all_sub_ses


def train_valid_test_split(data_path, out_dir, validation_size, override=False):
    train_path = os.path.join(out_dir, f'train-sub-ses.pth')
    valid_path = os.path.join(out_dir, f'valid-sub-ses.pth')
    test_path = os.path.join(out_dir, "test_sub_ses.pth")

    # sub450 to sub492 are voxel-wised segmented, so we choose them for test set
    with open(test_path, 'rb') as f:
        test_sub_ses = pickle.load(f)

    if os.path.isfile(train_path) and os.path.isfile(valid_path) and not override:
        with open(train_path, 'rb') as f1, open(valid_path, 'rb') as f2:
            train_sub_ses = pickle.load(f1)
            valid_sub_ses = pickle.load(f2)
    else:
        patients = find_sub_ses_pairs(os.path.join(data_path, "Positive_Patches"))
        patients = list(dict.fromkeys(patients))  # removing duplicates
        all_sub_ses = find_sub_ses_pairs(os.path.join(data_path, "Negative_Patches"))
        controls = [x for x in all_sub_ses if x not in patients]
        assert len(all_sub_ses) == len(patients) + len(controls), "the length of all sub-sessions is not equal to controls and patients"

        train_valid_patients = [x for x in patients if x not in test_sub_ses]
        train_valid_controls = [x for x in controls if x not in test_sub_ses]
        y = np.zeros(len(train_valid_patients) + len(train_valid_controls))
        y[:len(train_valid_patients)] = 1
        x = np.concatenate([np.array(train_valid_patients), np.array(train_valid_controls)])

        train_sub_ses, valid_sub_ses = train_test_split(x, test_size=validation_size, random_state=42, stratify=y)

        with open(train_path, 'wb') as f1, open(valid_path, 'wb') as f2:
            pickle.dump(train_sub_ses, f1)
            pickle.dump(valid_sub_ses, f2)

    return train_sub_ses, valid_sub_ses, test_sub_ses


class AneurysmDataset(Dataset):
    def __init__(self, root_dir, sub_ses_to_use, transform=None, shuffle=True):
        self.images_files = []
        self.labels = []
        self.masks_files = []
        self.transform = transform
        self.normalize = NormalizeIntensity()

        self.image_affine = None
        self.mask_affine = None
        self.read_dataset(root_dir, sub_ses_to_use)
        if shuffle:
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
                        if self.image_affine is None or self.mask_affine is None:
                            self.image_affine = nib.load(patch_path).affine
                            self.mask_affine = nib.load(mask_path).affine

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

        image = image.unsqueeze(0)  # to add a channel -> ch, h, w, d
        mask = mask.unsqueeze(0)  # to add a channel -> ch, h, w, d
        if self.transform is not None:
            if self.labels[item] != 0:
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)

        image = self.normalize(image)
        return image, mask, label


class CustomBatch:
    def __init__(self, data, only_label: bool):
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
