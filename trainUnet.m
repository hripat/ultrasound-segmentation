clear all;
% Set up Directories for Training Images and Labels
segDir = fullfile('C:\','Users','CompAST','Desktop','Hridayi','Seg_Binary'); % Segmentations 
USDir = fullfile('C:\','Users','CompAST','Desktop','Hridayi','USRename'); % Labels = US Images
imds = imageDatastore(USDir); %DataStore of input training images - ultrasound images
classNames = ["bone","background"]; %labels
labelIDs   = [1 0];
pxds = pixelLabelDatastore(segDir,classNames,labelIDs); %Has Ground Truth pixel data on training images  - Seg images are pixel labeled (only have 255/0)
% C = read(pxds)

%Create Unet
imageSize = [256 256 1]; %Size of images being used
numClasses = 2;  %Binary images so 2
encoderDepth = 4; %How many layers will there be in Unet apart from input and output
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth) %Create the Unet
plot(lgraph) 

options = trainingOptions('adam','InitialLearnRate', 3e-4, ...
    'MaxEpochs',100,'MiniBatchSize',15, ...
    'Plots','training-progress','Shuffle','every-epoch'); %Options for Training

%  'LearnRateSchedule', 'piecewise','LearnRateDropFactor',0.02,'LearnRateDropPeriod',5, ...

ds = pixelLabelImageDatastore(imds,pxds) %returns a datastore based on input image data(imds - US images) 
%and pxds (required network output - segmentations)

% imageAugmenter = imageDataAugmenter( ...
%     'RandXTranslation',[-10 10], ...
%     'RandYTranslation',[-10 10])
% 
% augimds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',imageAugmenter,'OutputSize',imageSize)

trainedNet_1600T11 = trainNetwork(ds,lgraph,options)
save trainedNet_1600T11

