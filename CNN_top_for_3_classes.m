Window = 500;
rng('default')
trainFlag = randperm(size(Input_train_split_data,2),62432*2);
Input_train_split_data = Input_train_split_data(trainFlag);
Target_train_split_data = new_3_class_label;
Target_train_split_data = Target_train_split_data(:,trainFlag);

Input_test_split_data = Input_test_split_data;
Target_test_split_data = new_3_class_label_test;


XTrain = zeros(Window,2,1,numel(Input_train_split_data));
a = cell2mat(Input_train_split_data); 
b = zeros(Window,2,1,numel(Input_train_split_data));
for i = 1:numel(Input_train_split_data)
    b(:,:,1,i) = a(:,((i-1)*2+1):i*2);
end
XTrain = b;
a = [];
b = [];


XTest = zeros(Window,2,1,numel(Input_test_split_data));
a = cell2mat(Input_test_split_data); 
b = zeros(Window,2,1,numel(Input_test_split_data));
for i = 1:numel(Input_test_split_data)
    b(:,:,1,i) = a(:,((i-1)*2+1):i*2);
end
XTest = b;


YTrain = zeros(size(Target_train_split_data,2),1);
for i = 1:size(Target_train_split_data,2)
    YTrain(i) = find(Target_train_split_data(:,i) == 1);
end
YTrain = categorical(YTrain);

YTest = zeros(size(Target_test_split_data,2),1);
for i = 1:size(Target_test_split_data,2)
    YTest(i) = find(Target_test_split_data(:,i) == 1);
end
YTest = categorical(YTest);

imageSize = size(XTrain);

imageSize = imageSize(1:3);

layers = [
    imageInputLayer(imageSize)

    convolution2dLayer([2 2],50)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],50)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
   

    convolution2dLayer([10 1],100)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],30)
    batchNormalizationLayer
    reluLayer
  
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    fullyConnectedLayer(3)
    %dropoutLayer(0.1)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer]


options = trainingOptions('adam',...
    'MaxEpochs',60, ...
    'Verbose',false,...
    'Plots','training-progress'); 
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',5, ...
    

%imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
%ds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);

cnn = trainNetwork(XTrain,YTrain,layers,options);
%cnn = trainNetwork(XTrain,YTrain,layers,options);
predictedLabels_train = classify(cnn,XTrain);
valLabels_train = YTrain;
accuracy_train_cnn = sum(predictedLabels_train == valLabels_train)/numel(valLabels_train)

predictedLabels = classify(cnn,XTest);                                              
valLabels = YTest;
accuracy_test_cnn = sum(predictedLabels == valLabels)/numel(valLabels)

%%
b1 = Target_test_raw_data;

d = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     d(i) = find(b1(:,i) == max(b1(:,i)));
 end
 
 d(find(d==1))=1;
 d(find(d==5))=1;
 d(find(d==7))=1;
 d(find(d==9))=1;
 
 d(find(d==2))=2;
 d(find(d==3))=2;
 d(find(d==8))=2;
 
 d(find(d==4))=3;
 d(find(d==6))=3;
 d(find(d==10))=3;
 
y = grp2idx(classify(cnn,XTest));
 
temp = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     temp(i) = mode(y((i-1)*1951+1:i*1951));
 end
 
accuracy_after_voting = size(find(temp == d),1)/size(d,1)

%save('cnn_best_top_100_new.mat','cnn','-v7.3','-nocompression')

% accuracy_train_cnn =
% 
%     0.9970
% 
% 
% accuracy_test_cnn =
% 
%     0.9498
% 
% 
% accuracy_after_voting =
% 
%      1

%Previous result:

% accuracy_train_cnn =
% 
%     0.9968
% 
% 
% accuracy_test_cnn =
% 
%     0.9450
% 
% 
% accuracy_after_voting =
% 
%     0.9938
