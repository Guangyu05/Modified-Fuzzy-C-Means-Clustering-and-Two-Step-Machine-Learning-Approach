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

    maxPooling2dLayer([3 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],50)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([3 1],'Stride',[2,1])
   

    convolution2dLayer([10 1],100)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([3 1],'Stride',[2,1])
    
    convolution2dLayer([10 1],30)
    batchNormalizationLayer
    reluLayer
  
    maxPooling2dLayer([3 1],'Stride',[2,1])
    
    fullyConnectedLayer(4)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer]


options = trainingOptions('adam',...
    'MaxEpochs',15, ...
    'Verbose',false,...
    'LearnRateSchedule','piecewise', ...
    'Plots','training-progress'); 

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

b1 = Target_raw_test;

d = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     d(i) = find(b1(:,i) == max(b1(:,i)));
 end
 
y = grp2idx(classify(cnn,XTest));
 
temp = zeros(size(b1,2),1);
 for i = 1:size(b1,2)
     temp(i) = mode(y((i-1)*1951+1:i*1951));
 end
 
accuracy_after_voting = size(find(temp == d),1)/size(d,1)


figure(1)
target_test = zeros(4,size(YTest,1));
for i = 1:size(YTest,1)
    target_test(YTest(i,:),i) = 1;
end

y = grp2idx(predictedLabels);
output_test = zeros(4,size(y,1));
for i = 1:size(y,1)
    output_test(y(i,:),i) = 1;
end
y=[];

plotconfusion(target_test,output_test)

figure(2)

target_train = zeros(4,size(YTrain,1));
for i = 1:size(YTrain,1)
    target_train(YTrain(i,:),i) = 1;
end

y = grp2idx(predictedLabels_train);
output_train = zeros(4,size(y,1));
for i = 1:size(y,1)
    output_train(y(i,:),i) = 1;
end

plotconfusion(target_train,output_train)

vwnn_in = activations(cnn,XTrain,16);

whos vwnn_in

% accuracy_train_cnn =
% 
%     0.9998
% 
% 
% accuracy_test_cnn =
% 
%     0.9638


% layers = 
% 
%   21x1 Layer array with layers:
% 
%      1   ''   Image Input             500x2x1 images with 'zerocenter' normalization
%      2   ''   Convolution             50 2x2 convolutions with stride [1  1] and padding [0  0  0  0]
%      3   ''   Batch Normalization     Batch normalization
%      4   ''   ReLU                    ReLU
%      5   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%      6   ''   Convolution             50 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%      7   ''   Batch Normalization     Batch normalization
%      8   ''   ReLU                    ReLU
%      9   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     10   ''   Convolution             100 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     11   ''   Batch Normalization     Batch normalization
%     12   ''   ReLU                    ReLU
%     13   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     14   ''   Convolution             30 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     15   ''   Batch Normalization     Batch normalization
%     16   ''   ReLU                    ReLU
%     17   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     18   ''   Fully Connected         4 fully connected layer
%     19   ''   Fully Connected         4 fully connected layer
%     20   ''   Softmax                 softmax
%     21   ''   Classification Output   crossentropyex
% 
% accuracy_train_cnn =
% 
%     0.9997
% 
% 
% accuracy_test_cnn =
% 
%     0.9728


% layers = 
% 
%   21x1 Layer array with layers:
% 
%      1   ''   Image Input             500x2x1 images with 'zerocenter' normalization
%      2   ''   Convolution             50 2x2 convolutions with stride [1  1] and padding [0  0  0  0]
%      3   ''   Batch Normalization     Batch normalization
%      4   ''   ReLU                    ReLU
%      5   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%      6   ''   Convolution             50 12x1 convolutions with stride [1  1] and padding [0  0  0  0]
%      7   ''   Batch Normalization     Batch normalization
%      8   ''   ReLU                    ReLU
%      9   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     10   ''   Convolution             100 12x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     11   ''   Batch Normalization     Batch normalization
%     12   ''   ReLU                    ReLU
%     13   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     14   ''   Convolution             30 12x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     15   ''   Batch Normalization     Batch normalization
%     16   ''   ReLU                    ReLU
%     17   ''   Max Pooling             3x1 max pooling with stride [2  1] and padding [0  0  0  0]
%     18   ''   Fully Connected         4 fully connected layer
%     19   ''   Fully Connected         4 fully connected layer
%     20   ''   Softmax                 softmax
%     21   ''   Classification Output   crossentropyex
% 
% accuracy_train_cnn =
% 
%     0.9999
% 
% 
% accuracy_test_cnn =
% 
%     0.9695
% 
%   Name          Size                          Bytes  Class     Attributes
% 
%   vwnn_in      42x1x30x249728            1258629120  single 
% 
% layers = 
% 
%   21x1 Layer array with layers:
% 
%      1   ''   Image Input             500x2x1 images with 'zerocenter' normalization
%      2   ''   Convolution             50 2x2 convolutions with stride [1  1] and padding [0  0  0  0]
%      3   ''   Batch Normalization     Batch normalization
%      4   ''   ReLU                    ReLU
%      5   ''   Max Pooling             3x1 max pooling with stride [3  1] and padding [0  0  0  0]
%      6   ''   Convolution             50 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%      7   ''   Batch Normalization     Batch normalization
%      8   ''   ReLU                    ReLU
%      9   ''   Max Pooling             3x1 max pooling with stride [3  1] and padding [0  0  0  0]
%     10   ''   Convolution             100 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     11   ''   Batch Normalization     Batch normalization
%     12   ''   ReLU                    ReLU
%     13   ''   Max Pooling             3x1 max pooling with stride [3  1] and padding [0  0  0  0]
%     14   ''   Convolution             30 10x1 convolutions with stride [1  1] and padding [0  0  0  0]
%     15   ''   Batch Normalization     Batch normalization
%     16   ''   ReLU                    ReLU
%     17   ''   Max Pooling             3x1 max pooling with stride [3  1] and padding [0  0  0  0]
%     18   ''   Fully Connected         4 fully connected layer
%     19   ''   Fully Connected         4 fully connected layer
%     20   ''   Softmax                 softmax
%     21   ''   Classification Output   crossentropyex
% 
% accuracy_train_cnn =
% 
%     0.9996
% 
% 
% accuracy_test_cnn =
% 
%     0.9618
% 
%   Name         Size                         Bytes  Class     Attributes
% 
%   vwnn_in      5x1x30x249728            149836800  single           
