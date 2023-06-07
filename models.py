from __future__ import print_function, division

import numpy as np
import random

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
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import time
import os
import copy

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torchvision.models.vision_transformer import vit_b_16
from self_attention_cv import ResNet50ViT
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.efficient import ViT
from linformer import Linformer
import torch.nn.functional as F
import torchvision


class Ditill_ViT(nn.Module):
    def __init__(self, input_size=(224, 224), output_size=5):
        super(Ditill_ViT, self).__init__()
        num_ftrs = 1000  # Output from pre-trained model

        teacher = torchvision.models.mobilenet_v3_small(pretrained=True)

        v = DistillableViT(
            image_size=input_size[0],
            patch_size=16,
            num_classes=num_ftrs,
            dim=1024,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.distiller = DistillWrapper(
            student=v,
            teacher=teacher,
            temperature=3,           # temperature of distillation
            alpha=0.5,               # trade between main loss and distillation loss
            hard=False               # whether to use soft or hard distillation
        )
        self.base_model = v

        self.input_size = input_size
        self.output_size = output_size

        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.classifier = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(
        ), nn.Dropout(0.5),
            nn.Linear(512, output_size))

        # self.main_pre_output_layer, self.main_output_layer = self._create_layer_block(hidden_size, output_size, attention)

        self.log_dict = {'validation acc': [], 'validation f1': [
        ], 'validation prec': [], 'validation recall': [], 'validation auc': []}
        self.step_log_dict = {'validation acc': [], 'validation f1': [
        ], 'validation prec': [], 'validation recall': [], 'validation auc': []}
        self.best_state_dict = None
        self.best_f1 = 0

    @property
    def get_distiller(self):
        return self.distiller

    def forward(self, inputs):
        X = self.base_model(inputs)
        X = self.classifier(X)
        return X


class CNN_MC(nn.Module):
    def __init__(self, input_size=(224, 224), output_size=5, regression=False):
        '''
        Params:
            input_size: The input size of the image (H, W)
            output_size: Number of outputs from last layer (num of classes / 1 for regression)
            regression: Train as a regression problem.
                '''
        super(CNN_MC, self).__init__()
        self.regression = regression
        self.num_ftrs = 1000  # Output from pre-trained model

        # CNN Model for transfer learning
        self.base_model = torchvision.models.efficientnet_v2_s(pretrained=True)
        # self.base_model = ResNet50ViT(img_dim=input_size[0], pretrained_resnet=True,
        #                  blocks=6, num_classes=num_ftrs,
        #                  dim_linear_block=256)

        # Vision transformer model
        self.base_model = vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT, image_size=input_size[0])
        self.input_size = input_size
        self.output_size = output_size

        if(self.regression == True):
            self.output_size = 1  # Treat as regression problem so output size is 1

        # Get base model for training
        #self.base_model = self.get_effcient_transformer()
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.classifier = nn.Sequential(nn.Linear(self.num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(512, 512), nn.ReLU(
        ), nn.Dropout(0.5),
            nn.Linear(512, self.output_size))

        # self.main_pre_output_layer, self.main_output_layer = self._create_layer_block(hidden_size, output_size, attention)

        self.log_dict = {'validation acc': [], 'validation f1': [
        ], 'validation prec': [], 'validation recall': [], 'validation auc': []}
        self.step_log_dict = {'validation acc': [], 'validation f1': [
        ], 'validation prec': [], 'validation recall': [], 'validation auc': []}
        self.best_state_dict = None
        self.best_f1 = 0

    def forward(self, inputs):
        X = self.base_model(inputs)
        X = self.classifier(X)
        return X

    @property
    def classes(self):
        return [0, 1, 2, 3, 4, 5]

    def get_effcient_transformer(self):

        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )

        model = ViT(
            dim=128,
            image_size=self.input_size[0],
            patch_size=32,
            num_classes=self.output_size,
            transformer=efficient_transformer,
            channels=3,
        )
        return model


# LSTM Model


class CNN_LSTM(nn.Module):
    def __init__(self, output_size=1, lstm_layers=1, regression=False):
        '''
                Params:
                        output_size: The number of outputs from the final MLP (num of classes / 1 for regression)
                        lstm_layers: Number of hidden layers in LSTM.
                        regression: Train as a regression problem.
        '''
        super(CNN_LSTM, self).__init__()
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.prediction_converter = None
        self.model = torchvision.models.resnet18(pretrained=True).to(device)

    def set_prediction_converter(self, hidden_state_space):
        if(self.prediction_converter is not None):
            return

        self.prediction_converter = nn.Sequential(
            nn.Linear(hidden_state_space, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, inputs):
        # Features extracted from input image
        feature_space_output = self.model(inputs)

        # LSTM to combine feature encoding from above with feature encodings from past networks
        # The input dimension is the volume of the remaining 3d image after convolution and pooling in resnet
        lstm_input_dimensions = 1000  # Out features from resnet18 is 1000

        if lstm_input_dimensions > 1000:
            print("This seems very large. Perhaps you should add some more layers to your network, or increase the kernel size.")
        self.lstm = nn.LSTM(lstm_input_dimensions,
                            lstm_input_dimensions, self.lstm_layers).to(device)

        # Set the output dimensions for FC
        self.set_prediction_converter(lstm_input_dimensions)

        #lstm_output, _ = self.lstm(feature_space_output)
        # The linear layer that maps from hidden state space to prediction space
        final_output = self.prediction_converter(feature_space_output)

        return final_output


class MTL_ViTModule(nn.Module):
    '''
        Vision Transformer Model for Multi-task learning for joints. 
        Predict Erosion and JSN Scores togehter

        params: 


    '''

    def __init__(self, input_size=(224, 224), output_size=1, pretrained=True, frozen_feature_layers=False):
        super().__init__()

        resnet18 = models.resnet18(pretrained=pretrained)
        self.output_size = output_size
        in_features = 1000  # Output from pre-trained model

        self.is_frozen = frozen_feature_layers
        self.base_model = vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT, image_size=input_size[0])

        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        # self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = self.base_model
        # this is needed because, nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). therefore we use
        # nn.Sequential and since sequential doesnt accept lists, we
        # unpack all items and send them like this
        #self.features = nn.Sequential(*self.features)

        if frozen_feature_layers:
            self.freeze_feature_layers()

        # it helps with performance. you can play with it
        # create more layers, play/experiment with them.
        self.classifier = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(512, 512), nn.ReLU(
        ), nn.Dropout(0.5),
            nn.Linear(512, self.output_size))

        self.fc0 = nn.Sequential(
            nn.Linear(in_features, in_features), nn.Dropout(0.5))
        self.bn_pu = nn.BatchNorm1d(in_features, eps=1e-5)

        # our two new heads for 2 tasks Erosion and JSN prediction
        self.fc_erosion = nn.Linear(in_features, self.output_size)
        self.fc_jsn = nn.Linear(in_features,  self.output_size)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        output = self.bn_pu(F.relu(self.fc0(output)))
        # since color is multi label we should use sigmoid
        # but since we want a numerical stable one, we use
        # nn.BCEWithLogitsloss, as a loss which itself applies sigmoid
        # and thus accepts logits. so we wont use sigmoid here for that matter
        # its much stabler than sigmoid+BCE
        prd_erosion = self.fc_erosion(output)
        prd_jsn = self.fc_jsn(output)

        return prd_erosion, prd_jsn

    def _set_freeze_(self, status):
        for n, p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status

    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)


class MTL_ViTModule_Wrist(nn.Module):
    '''
        Vision Transformer Model for Multi-task learning for Wrist only. 
        Predict the scores of different joints from single wrist image.

        params: 


    '''

    def __init__(self, input_size=(224, 224), output_size=1, pretrained=True, frozen_feature_layers=False, score_type="E"):
        super().__init__()

        resnet18 = models.efficientnet_v2_m(pretrained=pretrained)
        self.output_size = output_size
        in_features = 1000  # Output from pre-trained model
        # Set the score type E = Erosion | J = JSN. The joints will vary based on the score type.
        self.score_type = score_type
        self.is_frozen = frozen_feature_layers
        # self.base_model = vit_b_16(weights = torchvision.models.ViT_B_16_Weights.DEFAULT, image_size=input_size[0])
        # self.base_model = resnet18
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        # self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = self.base_model
        # this is needed because, nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). therefore we use
        # nn.Sequential and since sequential doesnt accept lists, we
        # unpack all items and send them like this
        #self.features = nn.Sequential(*self.features)

        if frozen_feature_layers:
            self.freeze_feature_layers()

        self.fc0 = nn.Sequential(
            nn.Linear(in_features, in_features), nn.Dropout(0.5))
        self.bn_pu = nn.BatchNorm1d(in_features, eps=1e-5)

        if(score_type == "E"):
            # Our four new heads for 4 joint erosion scores
            self.lunate_fc = nn.Linear(in_features, self.output_size)
            self.mc1_fc = nn.Linear(in_features,  self.output_size)
            self.mul_fc = nn.Linear(in_features,  self.output_size)
            self.nav_fc = nn.Linear(in_features,  self.output_size)
        if(score_type == "J"):
            # Our four new heads for 4 joint erosion scores
            self.capnlun_fc = nn.Linear(in_features, self.output_size)
            self.cmc3_fc = nn.Linear(in_features,  self.output_size)
            self.cmc4_fc = nn.Linear(in_features,  self.output_size)
            self.cmc5_fc = nn.Linear(in_features,  self.output_size)
            self.mna_fc = nn.Linear(in_features,  self.output_size)
            self.radcar_fc = nn.Linear(in_features,  self.output_size)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        output = self.bn_pu(F.relu(self.fc0(output)))
        # since color is multi label we should use sigmoid
        # but since we want a numerical stable one, we use
        # nn.BCEWithLogitsloss, as a loss which itself applies sigmoid
        # and thus accepts logits. so we wont use sigmoid here for that matter
        # its much stabler than sigmoid+BCE
        if(self.score_type == "E"):
            # Our four new heads for 4 joint erosion scores
            pred_lun = self.lunate_fc(output)
            pred_mc1 = self.mc1_fc(output)
            pred_mul = self.mul_fc(output)
            pred_nav = self.nav_fc(output)

            return pred_lun, pred_mc1, pred_mul, pred_nav

        if(self.score_type == "J"):
            # Our four new heads for 4 joint erosion scores
            pred_cap = self.capnlun_fc(output)
            pred_cmc3 = self.cmc3_fc(output)
            pred_cmc4 = self.cmc4_fc(output)
            pred_cmc5 = self.cmc5_fc(output)
            pred_mna = self.mna_fc(output)
            pred_rad = self.radcar_fc(output)

            return pred_cap, pred_cmc3, pred_cmc4, pred_cmc5, pred_mna, pred_rad

    def _set_freeze_(self, status):
        for n, p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status

    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)
