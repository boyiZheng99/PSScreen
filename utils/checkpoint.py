import os
import gc
import shutil

import torch


def load_pretrained_model(model, args):
    
    modelDict = model.backbone.state_dict()
    
    pretrainedModel = torch.load(args.pretrainedModel)
    if 'model' in pretrainedModel:
        pretrainedModel = pretrainedModel['model']
    pretrainedDict = {}
    for k,v in pretrainedModel.items():
        if k.startswith('fc') or  k.startswith('head'):
            continue
        pretrainedDict[k] = v
    modelDict.update(pretrainedDict)
    model.backbone.load_state_dict(modelDict, strict=False)

    del pretrainedModel
    del pretrainedDict
    gc.collect()

    return model


def save_checkpoint(args, state, isBest):

    outputPath = os.path.join('./exp/checkpoint/', args.post)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if isBest:
        torch.save(state, os.path.join(outputPath, 'Checkpoint_Best.pth'))
