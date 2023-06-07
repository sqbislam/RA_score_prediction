
import numpy as np
import random
from sklearn.preprocessing import binarize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
import math
import os
import copy

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from utils import save_pickled_file, load_pickled_file

class2idx = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

idx2class = {v: k for k, v in class2idx.items()}


class ObtainJointImagesData():
    '''
        Generate the Erosion and JSN score data of individual joint images.
        Params:
            score_type = Erosion or JSN score label (E / J)
            filter_joint = Type of joint to use. None means all joints are used. (mcp, pip, radius, ulna, wrist)
            test_size = Percentage of test set.
            binarize = Binarize the image labels. If Score > 0 then Damage else No Damage. For Classification

    '''

    def __init__(self, score_type="E", filter_joint=None, filter_score=None, test_size=0.35, binarize=False):
        self.data = None
        self.training_data = None
        self.test_data = None
        self.test_size = test_size
        self.score_type = score_type
        self.filter_score = filter_score
        self.filter_joint = filter_joint
        self.binarize = binarize
        self.__get_data()  # Prepare data

    def __get_data(self, csv_dir="./csvs/Joint_images_Feature_extracted_GLCM_AllScores.xlsx", img_dir="./RA_data/extracted_joint_RA_Jun2/"):
        filenames = []
        e_scores = []
        j_scores = []
        binarized_scores = []

        jdf = pd.read_excel(csv_dir, index_col=0).replace({"F": 0, "C": 0})

        for index, row in jdf.iterrows():
            erosion = row["Erosion Score"]
            jsn = row["JSN_Score"]

            filename = row["Filename"]
            joint_type = row["Joint_Type"]
            isWristRadiusUlna = "wrist" in joint_type.lower(
            ) or "radius" in joint_type.lower() or "ulna" in joint_type.lower()
            notZero = erosion != 0.0 and jsn != 0.0

            filter = True

            # If specific joint images are needed. Filter accordingly
            hasMCP = "mcp" in joint_type.lower()
            hasPIP = "pip" in joint_type.lower()
            hasRadius = "radius" in joint_type.lower()
            hasUlna = "ulna" in joint_type.lower()
            hasWrist = "wrist" in joint_type.lower()

            if(self.filter_joint == None):
                filter = True
            elif("mcp" in self.filter_joint.lower()):
                filter = hasMCP
            elif("pip" in self.filter_joint.lower()):
                filter = hasPIP
            elif("radius" in self.filter_joint.lower()):
                filter = hasRadius
            elif("ulna" in self.filter_joint.lower()):
                filter = hasUlna
            elif("wrist" in self.filter_joint.lower()):
                filter = hasWrist

            # Filter any given score value
            if(self.filter_score != None and erosion == self.filter_score):
                filter = False

            if(not np.isnan(erosion) and filter):
                # Add binary label (no damage) when both erosion and jsn are 0
                if(erosion <= 0 and not np.isnan(jsn) and jsn <= 0):
                    binarized_scores.append(0)
                else:
                    binarized_scores.append(1)

                e_scores.append(erosion)
                if(not np.isnan(jsn)):
                    j_scores.append(jsn)

                filenames.append(f"{img_dir}{row['Filename']}")

        # Choose outcome (either erosion scores or jsn scores)
        y = np.array(j_scores, dtype=np.int16)
        if(self.score_type == "E"):
            y = np.array(e_scores, dtype=np.int16)

        # If classificaiton based on binarized data then labels are binary scores
        if(self.binarize == True):
            y = binarized_scores

        # If multi-task score with both Erosion and JSN as labels
        if(self.score_type == "Both"):
            temp = np.array([[e, j] for e, j in zip(e_scores, j_scores)])
            y = temp

        y = torch.from_numpy(np.array(y)).long()

        self.data = {'X': filenames, 'y': y}

        X_train, X_test, y_train, y_test = train_test_split(
            filenames, y, test_size=self.test_size, random_state=42)
        self.training_data = (X_train, y_train)
        self.test_data = (X_test, y_test)

    def __get_processed_data(self, df, img_dir="./RA_data/extracted_joint_RA_Jun2/"):
        filenames = []
        e_scores = []
        j_scores = []
        for index, row in df.iterrows():
            erosion = row["Erosion Score"]
            jsn = row["JSN_Score"]

            filename = row["Filename"]
            joint_type = row["Joint_Type"]

            filter = True

            # If specific joint images are needed. Filter accordingly
            hasMCP = "mcp" in joint_type.lower()
            hasPIP = "pip" in joint_type.lower()
            hasRadius = "radius" in joint_type.lower()
            hasUlna = "ulna" in joint_type.lower()
            hasWrist = "wrist" in joint_type.lower()

            if(self.filter_joint == None):
                filter = True
            elif("mcp" in self.filter_joint.lower()):
                filter = hasMCP
            elif("pip" in self.filter_joint.lower()):
                filter = hasPIP
            elif("radius" in self.filter_joint.lower()):
                filter = hasRadius
            elif("ulna" in self.filter_joint.lower()):
                filter = hasUlna
            elif("wrist" in self.filter_joint.lower()):
                filter = hasWrist

            # Filter any given score value
            if(self.filter_score != None and erosion == self.filter_score):
                filter = False

            if(not np.isnan(erosion) and filter):

                e_scores.append(erosion)
                if(not np.isnan(jsn)):
                    j_scores.append(jsn)

                filenames.append(f"{img_dir}{row['Filename']}")

        # Choose outcome (either erosion scores or jsn scores)
        y = np.array(j_scores, dtype=np.int16)
        if(self.score_type == "E"):
            y = np.array(e_scores, dtype=np.int16)

        # If multi-task score with both Erosion and JSN as labels
        if(self.score_type == "Both"):
            temp = np.array([[e, j] for e, j in zip(e_scores, j_scores)])
            y = temp

        y = torch.from_numpy(np.array(y)).long()

        return {'X': filenames, 'y': y}

    def get_patient_split_data(self):

        train_df = pd.read_excel(
            "./csvs/Joint_Images_Train.xlsx", index_col=0).replace({"F": 0, "C": 0})
        val_df = pd.read_excel(
            "./csvs/Joint_Images_Val.xlsx", index_col=0).replace({"F": 0, "C": 0})
        test_df = pd.read_excel(
            "./csvs/Joint_Images_Test.xlsx", index_col=0).replace({"F": 0, "C": 0})

        train_set = self.__get_processed_data(train_df)
        val_set = self.__get_processed_data(val_df)
        test_set = self.__get_processed_data(test_df)
        return {'train': train_set,
                'val': val_set,
                'test': test_set}

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.test_data

    def get_data(self):
        return self.data


class ObtainWristJointImagesData():
    '''
        Generate the Multi-task data for wrist joints.
        Params:
            score_type = Erosion or JSN score label (E / J / Both) 
            filter_joint = Type of joint to use. None means all joints are used. (mcp, pip, radius, ulna, wrist)
            test_size = Percentage of test set.
            binarize = Binarize the image labels. If Score > 0 then Damage else No Damage. For Classification

    '''

    def __init__(self, score_type="E", filter_joint=None, test_size=0.35, binarize=False):
        self.data = None
        self.training_data = None
        self.test_data = None
        self.test_size = 0.35
        self.score_type = score_type
        self.filter_joint = filter_joint
        self.binarize = binarize
        self.__get_data()  # Prepare data

    def __get_data(self, img_dir="./RA_data/extracted_joint_RA_Jun2/"):
        csv_dir_e = "./RA_data/wrist_multijoint_erosion_scores_missingRemoved.xlsx"
        csv_dir_j = "./RA_data/wrist_multijoint_jsn_scores_missingRemoved.xlsx"

        EROSION_JOINTS_OF_INTEREST = ['lunate', 'mc1', 'mul', 'nav']
        JSN_JOINTS_OF_INTEREST = ['capnlun',
                                  'cmc3', 'cmc4', 'cmc5', 'mna', 'radcar']

        e_scores = []
        j_scores = []

        e_df = pd.read_excel(csv_dir_e, index_col=0)
        j_df = pd.read_excel(csv_dir_j, index_col=0)

        TRAIN_PATIENT_SET = load_pickled_file(
            "./csvs/patients_train_test_split")['trainval']
        TEST_PATIENT_SET = load_pickled_file(
            "./csvs/patients_train_test_split")['test']

        X_train, X_test, y_train, y_test = [], [], [], []

        # GET TRAIN SET
        for pid in TRAIN_PATIENT_SET:
            interested_df_e = e_df[e_df["PatientID"] == pid]
            interested_df_j = j_df[j_df["PatientID"] == pid]

            erosion_df = []
            # Find joints of interest
            for jn in EROSION_JOINTS_OF_INTEREST:
                erosion_df.append(
                    interested_df_e[interested_df_e["Joint"] == jn].values)

            a, b, c, d = erosion_df[0], erosion_df[1], erosion_df[2], erosion_df[3]
            for w, x, y, z in zip(a, b, c, d):
                X_train.extend([f'{img_dir}{w[2]}'])
                e_scores.extend([[w[4], x[4], y[4], z[4]]])

        # Choose outcome (either erosion scores or jsn scores)
        y_train = np.array(j_scores, dtype=np.int16)
        if(self.score_type == "E"):
            y_train = np.array(e_scores, dtype=np.int16)

        y_train = torch.from_numpy(np.array(y_train))

        # GET TEST SET

        e_scores = []
        j_scores = []
        for pid in TEST_PATIENT_SET:
            interested_df_e = e_df[e_df["PatientID"] == pid]
            interested_df_j = j_df[j_df["PatientID"] == pid]

            erosion_df = []
            # Find joints of interest
            for jn in EROSION_JOINTS_OF_INTEREST:
                erosion_df.append(
                    interested_df_e[interested_df_e["Joint"] == jn].values)

            a, b, c, d = erosion_df[0], erosion_df[1], erosion_df[2], erosion_df[3]
            for w, x, y, z in zip(a, b, c, d):
                X_test.extend([f'{img_dir}{w[2]}'])
                e_scores.extend([[w[4], x[4], y[4], z[4]]])

        # Choose outcome (either erosion scores or jsn scores)
        y_test = np.array(j_scores, dtype=np.int16)
        if(self.score_type == "E"):
            y_test = np.array(e_scores, dtype=np.int16)

        y_test = torch.from_numpy(np.array(y_test))

        self.data = {'X_train': X_train, 'y_train': y_train,
                     'X_test': X_test, 'y_test': y_test}

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.test_data

    def get_data(self):
        return self.data


class JointImageDataset(Dataset):
    '''
        Image dataset containing images of all joints of all types, removing timepoint and joint type info. 
        Outcome variable is score either of type Erosion or JSN
    '''

    def __init__(self, data, data_type, transform=None, target_transform=None, image_size=(224, 224)):
        # Get training data, images, erosion score and jsn scores
        filenames, score = data
        self.scores = score
        self.filenames = filenames
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomAutocontrast(p=0.7),
                transforms.RandomAdjustSharpness(1.4),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation(degrees=(-5, 5)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]),
        }
        self.target_transform = target_transform
        self.type = data_type

    def __len__(self):
        return len(self.filenames)

    @property
    def classes(self):
        return np.unique(self.scores)

    def __getitem__(self, idx):
        # Read image from filename
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")

        label_e, label_j = self.scores[idx]

        if self.data_transforms:
            image = self.data_transforms[self.type](image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


'''
    Dataset for multi-task learning of both scores combined
'''


class JointImageDatasetMultiTask(Dataset):
    '''
        Image dataset containing images of all joints of all types, removing timepoint and joint type info. 
        Outcome variable is combined score Erosion & JSN
    '''

    def __init__(self, data, data_type, transform=None, target_transform=None, image_size=(224, 224), isWrist=False):
        # Get training data, images, erosion score and jsn scores
        filenames, score = data
        self.scores = score
        self.filenames = filenames
        self.isWrist = isWrist
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomAutocontrast(p=0.7),
                transforms.RandomAdjustSharpness(1.4),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation(degrees=(-5, 5)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]),
        }
        self.target_transform = target_transform
        self.type = data_type

    def __len__(self):
        return len(self.filenames)

    @property
    def classes(self):
        return np.unique(self.scores)

    def __getitem__(self, idx):

        # Read image from filename
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.data_transforms:
            image = self.data_transforms[self.type](image)
        if(self.isWrist == True):
            return (image, *self.scores[idx])
        else:
            label_e, label_j = self.scores[idx]

            if self.target_transform:
                label_e = self.target_transform(label_e)
                label_j = self.target_transform(label_j)

            return image, label_e, label_j
