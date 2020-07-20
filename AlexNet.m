clear;
prompt='Enter the number of objects you want to classify';
titleBar = 'Initialization';
defaultString = 'Dont leave this empty';
n= (inputdlg(prompt, titleBar, 1, {defaultString}));
 
n=str2double(n);
cam=webcam(1);
rmdir('C:\Users\DELL\Documents\MATLAB\Examples\nnet\ClassifyImagesFromWebcamUsingDeepLearningExample\myimages','s');
pause;
for k=1:n
    prompt='Enter the name of the object you have placed';
titleBar = 'Object training';
defaultString = 'Dont leave this empty';
a= char(inputdlg(prompt, titleBar, 1, {defaultString}));
  
  %a=input(prompt,'s');
  dr='C:\Users\DELL\Documents\MATLAB\Examples\nnet\ClassifyImagesFromWebcamUsingDeepLearningExample\myimages\';
  
folder=strcat(dr,a)
  mkdir(folder);
  nametemplate='image_%04d.jpg'
  imnum=0;
   for K=1:10
     img=snapshot(cam);
     imnum=imnum+1;
     thisfile=sprintf(nametemplate,imnum);
     fullname=fullfile(folder,thisfile);
     imwrite(img,fullname);
 end
end
net=alexnet;
layers=net.Layers
layers(23)=fullyConnectedLayer(n);
layers(end)=classificationLayer
allImages = imageDatastore('C:\Users\DELL\Documents\MATLAB\Examples\nnet\ClassifyImagesFromWebcamUsingDeepLearningExample\myimages\', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
inputSize = [227 227];
allImages.ReadFcn = @(loc)imresize(imread(loc),inputSize);

[trainingImages, testImages] = splitEachLabel(allImages,1, 'randomize');
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 2, 'MiniBatchSize', 64);
myNet = trainNetwork(trainingImages, layers, opts);


inputSize=net.Layers(1).InputSize(1:2)
h = figure;
while ishandle(h)
    im = snapshot(cam);
    image(im)
    im = imresize(im,inputSize);
    [label,score] = classify(myNet,im);
    title({char(label), num2str(max(score),2)});
    drawnow
    
end



