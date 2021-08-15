run data_process_3_noise_new.m

Feature_Input_split_train = zeros(4,size(Input_train_split_data,2));

for i = 1:size(Input_train_split_data,2)
    Feature_Input_split_train(1,i) = sum(abs(Input_train_split_data{i}(:,1)));
    Feature_Input_split_train(2,i) = sum(abs(Input_train_split_data{i}(:,2)));
    Feature_Input_split_train(3,i) = std(abs(fft(Input_train_split_data{i}(:,1))));
    Feature_Input_split_train(4,i) = std(abs(fft(Input_train_split_data{i}(:,2))));
end


Feature_Input_split_test = zeros(4,size(Input_test_split_data,2));

for i = 1:size(Input_test_split_data,2)
    Feature_Input_split_test(1,i) = sum(abs(Input_test_split_data{i}(:,1)));
    Feature_Input_split_test(2,i) = sum(abs(Input_test_split_data{i}(:,2)));
    Feature_Input_split_test(3,i) = std(abs(fft(Input_test_split_data{i}(:,1))));
    Feature_Input_split_test(4,i) = std(abs(fft(Input_test_split_data{i}(:,2))));
end


%% The code below is for training KNN and RF as top classifier, date:19/01/2020
rng('default')
trainFlag = randperm(size(Feature_Input_split_train,2),62432*2);
XTrain = Feature_Input_split_train(:,trainFlag);
Target_train_split_data = new_3_class_label;
YTrain = Target_train_split_data(:,trainFlag);
d = zeros(size(YTrain,2),1);
 for i = 1:size(YTrain,2)
     d(i) = find(YTrain(:,i) == max(YTrain(:,i)));
 end
YTrain = d;
d=[];


XTest = Feature_Input_split_test;
YTest = new_3_class_label_test;

 d = zeros(size(YTest,2),1);
 for i = 1:size(YTest,2)
     d(i) = find(YTest(:,i) == max(YTest(:,i)));
 end

YTest = d;




