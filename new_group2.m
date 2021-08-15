%clear
%% I class(output=2)
s1i=csvread('S1-I.csv');
s1i=normalisation(s1i);
s2i=csvread('S2-I.csv');
s2i=normalisation(s2i);
%s3i=csvread('S3-I.csv');
%s3ift=extractfeatures(normalisation(s3i));
s4i=csvread('S4-I.csv');
s4i=normalisation(s4i);
s5i=csvread('S5-I.csv');
s5i=normalisation(s5i);
s6i=csvread('S6-I.csv');
s6i=normalisation(s6i);
%s7i=csvread('S7-I.csv');
%s7ift=extractfeatures(normalisation(s7i));
s8i=csvread('S8-I.csv');
s8i=normalisation(s8i);
s9i=csvread('S9-I.csv');
s9i=normalisation(s9i);
s10i=csvread('S10-I.csv');
s10i=normalisation(s10i);

I = [s1i,s2i,s4i,s5i,s6i,s8i,s9i,s10i];
[I_train,I_test,I_train_raw,I_test_raw,I_train_raw_target,...
 I_test_raw_target,I_train_split,I_test_split,I_train_split_target,...
 I_test_split_target] = EMGdataprocess_group_2_2(I,1);

%% L class(output=3)
s1l=csvread('S1-L.csv');
s1l=normalisation(s1l);
s2l=csvread('S2-L.csv');
s2l=normalisation(s2l);
%s3l=csvread('S3-L.csv');
%s3lft=extractfeatures(normalisation(s3l));
s4l=csvread('S4-L.csv');
s4l=normalisation(s4l);
s5l=csvread('S5-L.csv');
s5l=normalisation(s5l);
s6l=csvread('S6-L.csv');
s6l=normalisation(s6l);
%s7l=csvread('S7-L.csv');
%s7lft=extractfeatures(normalisation(s7l));
s8l=csvread('S8-L.csv');
s8l=normalisation(s8l);
s9l=csvread('S9-L.csv');
s9l=normalisation(s9l);
s10l=csvread('S10-L.csv');
s10l=normalisation(s10l);

L = [s1l,s2l,s4l,s5l,s6l,s8l,s9l,s10l];
[L_train,L_test,L_train_raw,L_test_raw,L_train_raw_target,...
 L_test_raw_target,L_train_split,L_test_split,L_train_split_target,...
 L_test_split_target] = EMGdataprocess_group_2_2(L,2);

%% TM class(output=8)
s1tm=csvread('S1-TM.csv');
s1tm=normalisation(s1tm);
s2tm=csvread('S2-TM.csv');
s2tm=normalisation(s2tm);
%s3tm=csvread('S3-TM.csv');
%s3tmft=extractfeatures(normalisation(s3tm));
s4tm=csvread('S4-TM.csv');
s4tm=normalisation(s4tm);
s5tm=csvread('S5-TM.csv');
s5tm=normalisation(s5tm);
s6tm=csvread('S6-TM.csv');
s6tm=normalisation(s6tm);
%s7tm=csvread('S7-TM.csv');
%s7tmft=extractfeatures(normalisation(s7tm));
s8tm=csvread('S8-TM.csv');
s8tm=normalisation(s8tm);
s9tm=csvread('S9-TM.csv');
s9tm=normalisation(s9tm);
s10tm=csvread('S10-TM.csv');
s10tm=normalisation(s10tm);

TM = [s1tm,s2tm,s4tm,s5tm,s6tm,s8tm,s9tm,s10tm];
[TM_train,TM_test,TM_train_raw,TM_test_raw,TM_train_raw_target,...
 TM_test_raw_target,TM_train_split,TM_test_split,TM_train_split_target,...
 TM_test_split_target] = EMGdataprocess_group_2_2(TM,3);



%%
Target_raw_test = [I_test_raw_target L_test_raw_target TM_test_raw_target];

Input_train_split_data = [I_train_split L_train_split TM_train_split];

Target_train_split_data = [I_train_split_target L_train_split_target TM_train_split_target];

Input_test_split_data = [I_test_split L_test_split TM_test_split];

Target_test_split_data = [I_test_split_target L_test_split_target TM_test_split_target];

function[inputnorm]=normalisation(input)
FlattenedInput = input(:)';
MappedFlattenedInput = mapminmax(FlattenedInput, 0, 1); 
inputnorm = reshape(MappedFlattenedInput, size(input));
end





