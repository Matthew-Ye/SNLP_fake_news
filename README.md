# SNLP_FakeNewsDetection

This is the project of SNLP course. We implement the fake news detection based on several machine learning algorithms and compare their results. It is mainly contained in the sourcefile *stance_detection.py*. We also fine tuned a RoBERTa model on our dataset. The sourcefile is *stance_RoBERTa.py*. 

Note that if you want fine tune the RoBERTa model with GPU. You need to check the official installment documentation carefully in [here](https://pytorch.org/). 

For us, we fine tuned the RoBERTa model on a Linux machine with CUDA 10.1. We installed the pytorch with command:
                
                pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

Other dependencies can be checked in the file *requirements.txt*

The project is based on the fake news challenge [FakeChallenge.org](http://fakenewschallenge.org).

The datasets are in the folder *fnc-1*.

The report is in the folder *report_files*.

This repository contains code that reads the dataset, extracts some simple features, trains a cross-validated model and performs an evaluation on a hold-out set of data. The results are also saved in the folder *results*.
