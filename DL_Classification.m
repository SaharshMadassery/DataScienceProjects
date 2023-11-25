clear
clc
%% 
% This assumes you have a directory: Scene_Categories % with each scene in a subdirectory
imds = imageDatastore('animals', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'ReadFcn', @readFunction);

%% Display Class Names and Counts
tbl = countEachLabel(imds); % Count the number of images per class
categories = tbl.Label;

%% Display a Sampling of Image Data
sample = splitEachLabel(imds, 16); % Split the data for display
montage(sample.Files(1:16)); 
title(char(tbl.Label(1)));

%% Show Sampling of All Data
for ii = 1:4
    sf = (ii - 1) * 16 + 1;
    ax(ii) = subplot(2, 2, ii);
    montage(sample.Files(sf:sf+3));
    title(char(tbl.Label(ii)));
end

%% Split the Data into Training and Validation Sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% Display a Subset of Training Images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4, 4, i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

%% Load the Pretrained AlexNet
net = alexnet;

%% Analyze the Network Architecture
analyzeNetwork(net)

%% Get the Input Size of the Network
inputSize = net.Layers(1).InputSize

%% Define Layers for Transfer Learning
layersTransfer = net.Layers(1:end-3);

%% Specify the Number of Classes
numClasses = 4

%% Define the Full Network Architecture
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%% Define Data Augmentation Parameters
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

%% Create Augmented Datastore for Training
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

%% Create Augmented Datastore for Validation
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% Define Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the Transfer Learning Network
netTransfer = trainNetwork(augimdsTrain, layers, options);

%% Classify Images in the Validation Set
[YPred, scores] = classify(netTransfer, augimdsValidation);

%% Display Randomly Selected Validation Images with Predicted Labels
idx = randperm(numel(imdsValidation.Files), 4);
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsValidation, idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% Calculate the Classification Accuracy
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

%% Define a Function to Read Images and Convert to Grayscale
function X = readFunctionConvertToGray(filename)
    X = imread(filename);
    if size(X, 3) == 3
        X = rgb2gray(X);
    end
end

%%% Define a Function to Read Images and Convert to RGB if Grayscale
function X = readFunction(filename)
    X = imread(filename);
    if size(X, 3) == 1  % If the image is grayscale
        X = cat(3, X, X, X);  % Convert it to RGB format with three channels
    end
end