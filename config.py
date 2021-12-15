import math
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

sweep_config = {
    'method': 'grid',
    'name':'grid-wrong_model_test',
    'metric' : {
        'name': 'best_acc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 30},
        'batch_size': {
            'value' : 25},
        'model': {
            'values': ['Wrong_resnet', 'resnet']}, #'values' : ['resnet', 'scratch'] 
        'optimizer': { 
            'value': 'adabelief'},#'values': ['adam', 'sgd', 'adabelief']
        'warm_up':{
            'value': 'no'}, #'values': ['yes', 'no']
        'seed':{
            'value': 0},#'values': [0, 3407]
        'learning_rate': {
            'value': 0.005},#'values': [0.001, 0.005]

    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }