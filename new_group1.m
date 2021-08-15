%clear;clc
%% HC class (output = 1)
s1hc=csvread('S1-HC.csv');
s1hc=normalisation(s1hc);
s2hc=csvread('S2-HC.csv');
s2hc=normalisation(s2hc);
%s3hc=csvread('S3-HC.csv');
%s3hcft=extractfeatures(normalisation(s3hc));
s4hc=csvread('S4-HC.csv');
s4hc=normalisation(s4hc);
s5hc=csvread('S5-HC.csv');
s5hc=normalisation(s5hc);
s6hc=csvread('S6-HC.csv');
s6hc=normalisation(s6hc);
%s7hc=csvread('S7-HC.csv');
%s7hcft=extractfeatures(normalisation(s7hc));
s8hc=csvread('S8-HC.csv');
s8hc=normalisation(s8hc);
s9hc=csvread('S9-HC.csv');
s9hc=normalisation(s9hc);
s10hc=csvread('S10-HC.csv');
s10hc=normalisation(s10hc);

HC = [s1hc s2hc s4hc s5hc s6hc s8hc s9hc s10hc];
[HC_train,HC_test,HC_train_raw,HC_test_raw,HC_train_raw_target,...
 HC_test_raw_target,HC_train_split,HC_test_split,HC_train_split_target,...
 HC_test_split_target] = EMGdataprocess_group_1(HC,1);

%% R class(output=5)
s1r=csvread('S1-R.csv');
s1r=normalisation(s1r);
s2r=csvread('S2-R.csv');
s2r=normalisation(s2r);
%s3r=csvread('S3-R.csv');
%s3rft=extractfeatures(normalisation(s3r));
s4r=csvread('S4-R.csv');
s4r=normalisation(s4r);
s5r=csvread('S5-R.csv');
s5r=normalisation(s5r);
s6r=csvread('S6-R.csv');
s6r=normalisation(s6r);
%s7r=csvread('S7-R.csv');
%s7rft=extractfeatures(normalisation(s7r));
s8r=csvread('S8-R.csv');
s8r=normalisation(s8r);
s9r=csvread('S9-R.csv');
s9r=normalisation(s9r);
s10r=csvread('S10-R.csv');
s10r=normalisation(s10r);

R = [s1r,s2r,s4r,s5r,s6r,s8r,s9r,s10r];
[R_train,R_test,R_train_raw,R_test_raw,R_train_raw_target,...
 R_test_raw_target,R_train_split,R_test_split,R_train_split_target,...
 R_test_split_target] = EMGdataprocess_group_1(R,2);

%% TL class(output=7)
s1tl=csvread('S1-TL.csv');
s1tl=normalisation(s1tl);
s2tl=csvread('S2-TL.csv');
s2tl=normalisation(s2tl);
%s3tl=csvread('S3-TL.csv');
%s3tlft=extractfeatures(normalisation(s3tl));
s4tl=csvread('S4-TL.csv');
s4tl=normalisation(s4tl);
s5tl=csvread('S5-TL.csv');
s5tl=normalisation(s5tl);
s6tl=csvread('S6-TL.csv');
s6tl=normalisation(s6tl);
%s7tl=csvread('S7-TL.csv');
%s7tlft=extractfeatures(normalisation(s7tl));
s8tl=csvread('S8-TL.csv');
s8tl=normalisation(s8tl);
s9tl=csvread('S9-TL.csv');
s9tl=normalisation(s9tl);
s10tl=csvread('S10-TL.csv');
s10tl=normalisation(s10tl);

TL = [s1tl,s2tl,s4tl,s5tl,s6tl,s8tl,s9tl,s10tl];
[TL_train,TL_test,TL_train_raw,TL_test_raw,TL_train_raw_target,...
 TL_test_raw_target,TL_train_split,TL_test_split,TL_train_split_target,...
 TL_test_split_target] = EMGdataprocess_group_1(TL,3);

%% TR class(output=9)
s1tr=csvread('S1-TR.csv');
s1tr=normalisation(s1tr);
s2tr=csvread('S2-TR.csv');
s2tr=normalisation(s2tr);
%s3tr=csvread('S3-TR.csv');
%s3trft=extractfeatures(normalisation(s3tr));
s4tr=csvread('S4-TR.csv');
s4tr=normalisation(s4tr);
s5tr=csvread('S5-TR.csv');
s5tr=normalisation(s5tr);
s6tr=csvread('S6-TR.csv');
s6tr=normalisation(s6tr);
%s7tr=csvread('S7-TR.csv');
%s7trft=extractfeatures(normalisation(s7tr));
s8tr=csvread('S8-TR.csv');
s8tr=normalisation(s8tr);
s9tr=csvread('S9-TR.csv');
s9tr=normalisation(s9tr);
s10tr=csvread('S10-TR.csv');
s10tr=normalisation(s10tr);

TR = [s1tr,s2tr,s4tr,s5tr,s6tr,s8tr,s9tr,s10tr];
[TR_train,TR_test,TR_train_raw,TR_test_raw,TR_train_raw_target,...
 TR_test_raw_target,TR_train_split,TR_test_split,TR_train_split_target,...
 TR_test_split_target] = EMGdataprocess_group_1(TR,4);

Target_raw_test = [HC_test_raw_target R_test_raw_target TL_test_raw_target TR_test_raw_target];

Input_train_split_data = [HC_train_split R_train_split TL_train_split TR_train_split];

Target_train_split_data = [HC_train_split_target R_train_split_target TL_train_split_target TR_train_split_target];

Input_test_split_data = [HC_test_split R_test_split TL_test_split TR_test_split];

Target_test_split_data = [HC_test_split_target R_test_split_target TL_test_split_target TR_test_split_target];


function[inputnorm]=normalisation(input)
FlattenedInput = input(:)';
MappedFlattenedInput = mapminmax(FlattenedInput, 0, 1); 
inputnorm = reshape(MappedFlattenedInput, size(input));
end


