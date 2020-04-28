# -*- coding: utf-8 -*-
"""
Fine tune RoBERTa model on FNC-1 dataset on stance detection task
"""

import sys
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
from utils.score import report_score, LABELS, score_submission, print_confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Stance Detection with RoBERTa')
parser.add_argument('--fromScratch', '-s', action='store_true', default=True, help="train from scratch or train from checkpoints")
parser.add_argument('--mode', '-m', type=str, default='train', help="train or test")
parser.add_argument("dir", type=str, default='models_RoBerta_1', nargs="?",
                    help="Directory where models and checkpoints are saved")
params = parser.parse_args()

train_from_scratch = params.fromScratch
output_dir = params.dir

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Loda datasets
train_bodies = pd.read_csv('fnc-1/train_bodies.csv') # 1683 bodies 
train_stances = pd.read_csv('fnc-1/train_stances.csv') # 49972 headlines and stances 
competition_test_stances = pd.read_csv('fnc-1/competition_test_stances.csv') # 25413 headlines and stances 
competition_test_bodies = pd.read_csv('fnc-1/competition_test_bodies.csv') # 904 bodies
# Join datasets
training_set = train_stances.join(train_bodies.set_index('Body ID'), on='Body ID')
competition_set = competition_test_stances.join(competition_test_bodies.set_index('Body ID'), on='Body ID')

labels_int = ['agree', 'disagree', 'discuss', 'unrelated']

training_set = pd.DataFrame(training_set.loc[:,['Headline', 'articleBody','Stance']])
training_set.columns = ['text_a', 'text_b', 'labels']
training_set["labels"] = training_set["labels"].apply(lambda x: labels_int.index(x))
train_df, val_df = train_test_split(training_set, random_state = 0)
competition_set = pd.DataFrame(competition_set.loc[:,['Headline', 'articleBody','Stance']])
competition_set.columns = ['text_a', 'text_b', 'labels']
competition_set["labels"] = competition_set["labels"].apply(lambda x: labels_int.index(x))
labels_test = list(competition_set["labels"])
from simpletransformers.classification import ClassificationModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if params.mode == 'train':
# Train model from scratch 
    if train_from_scratch:
        model = ClassificationModel('roberta', 'roberta-base', num_labels=4, use_cuda=True, args={
            'learning_rate':3e-5,
            'num_train_epochs': 10,
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'process_count': 10,
            # 'train_batch_size': 8,
            # 'eval_batch_size': 8,
            'max_seq_length': 512,
            'output_dir': 'models_RoBerta_1',
            'save_model_every_epoch': True,
            'tensorboard_dir': 'log_Roberta_1',
            'fp16': False,
            'save_steps': 4000,
        })
    else:
        # Load checkpoints and continue training
        model = ClassificationModel('roberta', 'models_RoBerta_1/checkpoint-37480-epoch-8/', num_labels=4, use_cuda=True, args={
            'learning_rate':3e-5,
            'num_train_epochs': 5,
            'reprocess_input_data': True,
            'overwrite_output_dir': False,
            'process_count': 10,
            # 'train_batch_size': 8,
            # 'eval_batch_size': 8,
            'max_seq_length': 512,
            'output_dir': 'models_continue',
            'save_model_every_epoch': True,
            'tensorboard_dir': 'log',
            'fp16': False,
        })        
    model.train_model(train_df)
else:
    # Load checkpoints and test
    model = ClassificationModel('roberta', 'models_RoBerta_1/checkpoint-37480-epoch-8/', num_labels=4, use_cuda=True, args={
        'learning_rate':3e-5,
        'num_train_epochs': 5,
        'reprocess_input_data': True,
        'overwrite_output_dir': False,
        'process_count': 10,
        # 'train_batch_size': 8,
        # 'eval_batch_size': 8,
        'max_seq_length': 512,
        'output_dir': 'models_continue',
        'save_model_every_epoch': True,
        'tensorboard_dir': 'log',
        'fp16': False,
    })

    

result, model_outputs, _ = model.eval_model(competition_set, acc=sklearn.metrics.accuracy_score)

preds_test = np.argmax(model_outputs, axis=1)

from sklearn.metrics import f1_score

def calculate_f1_scores(y_true, y_predicted):
    f1_macro = f1_score(y_true, y_predicted, average='macro')
    f1_classwise = f1_score(y_true, y_predicted, average=None, labels=[0, 1, 2, 3])

    resultstring = "F1 macro: {:.3f}".format(f1_macro * 100) + "% \n"
    resultstring += "F1 agree: {:.3f}".format(f1_classwise[0] * 100) + "% \n"
    resultstring += "F1 disagree: {:.3f}".format(f1_classwise[1] * 100) + "% \n"
    resultstring += "F1 discuss: {:.3f}".format(f1_classwise[2] * 100) + "% \n"
    resultstring += "F1 unrelated: {:.3f}".format(f1_classwise[3] * 100) + "% \n"

    return resultstring

f1_scores = calculate_f1_scores(preds_test, labels_test)
print(f1_scores)

#Run on competition dataset
predicted = [LABELS[int(a)] for a in preds_test]
actual = [LABELS[int(a)] for a in labels_test]

print("Scores on the test set")
report_score(actual,predicted)

eval_report = classification_report(labels_test, preds_test)
print('Test report', eval_report)
