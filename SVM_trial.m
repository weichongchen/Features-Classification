clc
clear all
close all

path = imageDatastore('./Assignment06_data_reduced/TrainingDataset/', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
classifier = trainImageCategoryClassifier(path,bagOfFeatures(path));

pathtest = imageDatastore('./Assignment06_data_reduced/TEST/', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
evaluate(classifier,pathtest);
   