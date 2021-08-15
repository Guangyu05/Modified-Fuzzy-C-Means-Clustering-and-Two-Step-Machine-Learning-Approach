run data_process_3_noise_new_1.m

%% Training labels
% New class 1
ind = ismember(Target_train_split_data',[1 0 0 0 0 0 0 0 0 0],'rows');
class1_1 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 1 0 0 0 0 0],'rows');
class1_5 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 0 0 1 0 0 0],'rows');
class1_7 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 0 0 0 0 1 0],'rows');
class1_9 = find(ind==1);

class1_new = [class1_1; class1_5; class1_7; class1_9];

% New class 2
ind = ismember(Target_train_split_data',[0 1 0 0 0 0 0 0 0 0],'rows');
class2_2 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 1 0 0 0 0 0 0 0],'rows');
class2_3 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 0 0 0 1 0 0],'rows');
class2_8 = find(ind==1);

class2_new = [class2_2; class2_3; class2_8];

% New class 3
ind = ismember(Target_train_split_data',[0 0 0 1 0 0 0 0 0 0],'rows');
class3_4 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 0 1 0 0 0 0],'rows');
class3_6 = find(ind==1);

ind = ismember(Target_train_split_data',[0 0 0 0 0 0 0 0 0 1],'rows');
class3_10 = find(ind==1);

class3_new = [class3_4; class3_6; class3_10];

new_3_class_label = zeros(3,size(Target_train_split_data,2));
new_3_class_label(1,class1_new) = 1;
new_3_class_label(2,class2_new) = 1;
new_3_class_label(3,class3_new) = 1;


%% Test labels
% New class 1
ind = ismember(Target_test_split_data',[1 0 0 0 0 0 0 0 0 0],'rows');
class1_1 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 1 0 0 0 0 0],'rows');
class1_5 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 0 0 1 0 0 0],'rows');
class1_7 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 0 0 0 0 1 0],'rows');
class1_9 = find(ind==1);

class1_new = [class1_1; class1_5; class1_7; class1_9];

% New class 2
ind = ismember(Target_test_split_data',[0 1 0 0 0 0 0 0 0 0],'rows');
class2_2 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 1 0 0 0 0 0 0 0],'rows');
class2_3 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 0 0 0 1 0 0],'rows');
class2_8 = find(ind==1);

class2_new = [class2_2; class2_3; class2_8];

% New class 3
ind = ismember(Target_test_split_data',[0 0 0 1 0 0 0 0 0 0],'rows');
class3_4 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 0 1 0 0 0 0],'rows');
class3_6 = find(ind==1);

ind = ismember(Target_test_split_data',[0 0 0 0 0 0 0 0 0 1],'rows');
class3_10 = find(ind==1);

class3_new = [class3_4; class3_6; class3_10];

new_3_class_label_test = zeros(3,size(Target_test_split_data,2));
new_3_class_label_test(1,class1_new) = 1;
new_3_class_label_test(2,class2_new) = 1;
new_3_class_label_test(3,class3_new) = 1;

