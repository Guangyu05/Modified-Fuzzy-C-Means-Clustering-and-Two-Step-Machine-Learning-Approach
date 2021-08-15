%Overlapped windowing
function [y_train,y_test,y_train_raw,y_test_raw,y_train_raw_target,...
    y_test_raw_target,y_train_split,y_test_split,y_train_split_target,...
    y_test_split_target] = EMGdataprocess_group_1(y,n)

Window = 500;
Incre = 10;
Datalength = size(y,1);


label = 1:96;
Train_label = [];
for i = 1:8
    Train_label = [Train_label label((i-1)*12+1:(i-1)*12+8)];
end
Test_label = setdiff(label,Train_label);      
        

y_train = y(:,Train_label);
y_test = y(:,Test_label);

y_train_raw = {};
for i = 0:size(Train_label,2)/2-1
        y_train_raw{i+1} = y_train(:,2*i+1:2*i+2);
end
y_train_raw_target = zeros(4,size(Train_label,2)/2);
y_train_raw_target(n,:) = ones(1,size(Train_label,2)/2);

y_test_raw = {};
for i = 0:size(Test_label,2)/2-1
    y_test_raw{i+1} = y_test(:,2*i+1:2*i+2);
end
y_test_raw_target = zeros(4,size(Test_label,2)/2);
y_test_raw_target(n,:) = ones(1,size(Test_label,2/2));

split_samples = (Datalength - Window)/Incre + 1;

y_train_split = {};
for i = 0:numel(y_train_raw)-1
    for j = 0:split_samples-1
        y_train_split{i*(split_samples)+j+1} = y_train_raw{i+1}(Incre*j+1:Incre*j+Window,:);
    end
end

y_train_split_target = zeros(4,numel(y_train_split));
y_train_split_target(n,:) = ones(1,numel(y_train_split));

y_test_split = {};
for i = 0:numel(y_test_raw)-1
    for j = 0:split_samples-1
        y_test_split{i*(split_samples)+j+1} = y_test_raw{i+1}(Incre*j+1:Incre*j+Window,:);
    end
end

y_test_split_target = zeros(4,numel(y_test_split));
y_test_split_target(n,:) = ones(1,numel(y_test_split));

end