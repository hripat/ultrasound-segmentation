clc
close all;
clear all
load trainedNet_1600T10
Net2 = trainedNet_1600T10; %renaming U-Net
testLabelDir = fullfile('C:\','Users','CompAST','Desktop','Hridayi','ValidationUS'); %Testing images - Original US 
validationDir = fullfile('C:\','Users','CompAST','Desktop','Hridayi','BinaryValidationSeg');  %Manual Segs of testing images

imdsTest = imageDatastore(testLabelDir);
imdsVal = imageDatastore(validationDir);

for i = 1:400 %looping through all 400 images in the validation set
    orig = readimage(imdsTest,i); %orig is US image
    man = readimage(imdsVal,i);% man is the manual segmentation
    man = logical(man); %converting manual image to binary image (thresholding)
    
    nn2 = predict(Net2,orig); %built-in function uses U-Net to predict locations, produces an image with decimal values from 0 to 1
    nn2b = imbinarize(nn2); %converts the 0 to 1 image to a binary image (thresholding > 0.5)
    nn2_new = nn2b(:,:,1); 
    dn2(i) = dice(man,nn2_new>0.6); %dice coefficient
    %bn2(i) = jaccard(man,nn2_new>0.5); %jaccard coefficient

end

dice = mean(dn2) % mean accuracy

im = readimage(imdsTest,3);
%imshow(im);
impred = predict(Net2,im);
pred = imbinarize(impred);
pred2 = pred(:,:,1);
figure;
%imshow(pred2)
imshowpair(im,pred2,'montage')

