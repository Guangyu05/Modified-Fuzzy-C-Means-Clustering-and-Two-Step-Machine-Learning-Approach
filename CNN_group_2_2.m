Window = 500;

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
    
    convolution2dLayer([8 1],50)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
   

    convolution2dLayer([8 1],100)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    convolution2dLayer([8 1],30)
    batchNormalizationLayer
    reluLayer
  
    maxPooling2dLayer([2 1],'Stride',[2,1])
    
    fullyConnectedLayer(3)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer]

% layers = [
%     imageInputLayer(imageSize)
% 
%     convolution2dLayer([2 2],50)
%     batchNormalizationLayer
%     reluLayer
%     %dropoutLayer(0.2)
% 
%     maxPooling2dLayer([2 1])
%     
%     convolution2dLayer([5 1],50)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer([2 1])
%    
% 
%     convolution2dLayer([5 1],100)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer([2 1])
%     
%     convolution2dLayer([5 1],50)
%     batchNormalizationLayer
%     reluLayer
%     %dropoutLayer
%   
%     fullyConnectedLayer(3)
%     fullyConnectedLayer(3)
%     softmaxLayer
%     classificationLayer]


options = trainingOptions('adam',...
    'MaxEpochs',10, ...
    'Verbose',false,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'Plots','training-progress'); 

%imageAugmenter = imageDataAugmenter('RandRotation',[-10 10]);
%ds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);

cnn_goup_3 = trainNetwork(XTrain,YTrain,layers,options);
%cnn = trainNetwork(XTrain,YTrain,layers,options);
predictedLabels_train = classify(cnn_goup_3,XTrain);
valLabels_train = YTrain;
accuracy_train_cnn = sum(predictedLabels_train == valLabels_train)/numel(valLabels_train)

predictedLabels = classify(cnn_goup_3,XTest);                                              
valLabels = YTest;
accuracy_test_cnn = sum(predictedLabels == valLabels)/numel(valLabels)

%%
b1 = Target_raw_test;

d = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     d(i) = find(b1(:,i) == max(b1(:,i)));
 end
 
y = grp2idx(classify(cnn_goup_3,XTest));
 
temp = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     temp(i) = mode(y((i-1)*1951+1:i*1951));
 end
 
accuracy_after_voting = size(find(temp == d),1)/size(d,1)

%%

target_test = zeros(3,size(YTest,1));
for i = 1:size(YTest,1)
    target_test(YTest(i,:),i) = 1;
end

y = grp2idx(predictedLabels);
output_test = zeros(3,size(y,1));
for i = 1:size(y,1)
    output_test(y(i,:),i) = 1;
end
y=[];

plotconfusion(target_test,output_test)

target_train = zeros(3,size(YTrain,1));
for i = 1:size(YTrain,1)
    target_train(YTrain(i,:),i) = 1;
end

y = grp2idx(predictedLabels_train);
output_train = zeros(3,size(y,1));
for i = 1:size(y,1)
    output_train(y(i,:),i) = 1;
end

plotconfusion(target_train,output_train)

