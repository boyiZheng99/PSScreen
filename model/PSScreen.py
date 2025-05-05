import numpy as np

import torch
import torch.nn as nn

from .backbone.Uresnet_SD import uresnet101
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer
import torch.nn.functional as F

class PSScreen(nn.Module):

    def __init__(self, wordFeatures,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=11, wordFeatureDim=300,alpha=0.95
                 ):

        super(PSScreen, self).__init__()

        self.backbone = uresnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim), )

        self.classNum = classNum

        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim

        self.wordFeatures = self.load_features(wordFeatures)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim,
                                                     intermediaDim=intermediaDim)

    

        self.classifiers = Element_Wise_Layer(classNum, outputDim)
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()
        self.fcSemantic = nn.Linear(imageFeatureDim, outputDim)


    def forward(self, input, target=None):

        
        featureMap = self.backbone(input, False)  # original features
        result, semanticFeature = self.process_feature_map(featureMap)

        if (not self.training):
            return result
            
        featureMap_aug = self.backbone(input)  # augmented features
        result_aug, semanticFeature_aug = self.process_feature_map(featureMap_aug)

        

        if target is None:
            return result, result_aug, semanticFeature, semanticFeature_aug
        
        pseudolabel = self.cal_pseudolabel(target,result)
        return result, result_aug, semanticFeature, semanticFeature_aug, pseudolabel
    

    def process_feature_map(self, feature_map):
        batchSize = feature_map.size(0)
        if feature_map.size(1) != self.imageFeatureDim:
            feature_map = self.changeChannel(feature_map)
        semantic_feature = self.SemanticDecoupling(feature_map, self.wordFeatures)[0]    #(batchSize, classNum, outputDim)
        outputSemantic = torch.tanh(self.fcSemantic(semantic_feature))                              
        result = self.classifiers(outputSemantic)  
        return result, semantic_feature

    def cal_pseudolabel(self,target,result):
        pseudolabel = torch.zeros_like(target)
        mask = torch.zeros_like(target)
        mask[~((target == 1) | (target == -1))] = 1
        result_clone = result.detach() #用来计算伪标签
        result_clone = self.sigmoid(result_clone)
        pseudolabel[result_clone > self.alpha] = 1
        pseudolabel[result_clone < 1-self.alpha] = -1
        pseudolabel = pseudolabel*mask
        return pseudolabel

    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)


