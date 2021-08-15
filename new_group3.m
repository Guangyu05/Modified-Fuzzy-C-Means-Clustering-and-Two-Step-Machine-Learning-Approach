%% M class(output=4)
s1m=csvread('S1-M.csv');
s1m=normalisation(s1m);
s2m=csvread('S2-M.csv');
s2m=normalisation(s2m);
%s3m=csvread('S3-M.csv');
%s3mft=extractfeatures(normalisation(s3m));
s4m=csvread('S4-M.csv');
s4m=normalisation(s4m);
s5m=csvread('S5-M.csv');
s5m=normalisation(s5m);
s6m=csvread('S6-M.csv');
s6m=normalisation(s6m);
%s7m=csvread('S7-M.csv');
%s7mft=extractfeatures(normalisation(s7m));
s8m=csvread('S8-M.csv');
s8m=normalisation(s8m);
s9m=csvread('S9-M.csv');
s9m=normalisation(s9m);
s10m=csvread('S10-M.csv');
s10m=normalisation(s10m);
M = [s1m,s2m,s4m,s5m,s6m,s8m,s9m,s10m];
[M_train,M_test,M_train_raw,M_test_raw,M_train_raw_target,...
 M_test_raw_target,M_train_split,M_test_split,M_train_split_target,...
 M_test_split_target] = EMGdataprocess_group_2_2(M,1);

%% TI class(output=6)
s1ti=csvread('S1-TI.csv');
s1ti=normalisation(s1ti);
s2ti=csvread('S2-TI.csv');
s2ti=normalisation(s2ti);
%s3ti=csvread('S3-TI.csv');
%s3tift=extractfeatures(normalisation(s3ti));
s4ti=csvread('S4-TI.csv');
s4ti=normalisation(s4ti);
s5ti=csvread('S5-TI.csv');
s5ti=normalisation(s5ti);
s6ti=csvread('S6-TI.csv');
s6ti=normalisation(s6ti);
%s7ti=csvread('S7-TI.csv');
%s7tift=extractfeatures(normalisation(s7ti));
s8ti=csvread('S8-TI.csv');
s8ti=normalisation(s8ti);
s9ti=csvread('S9-TI.csv');
s9ti=normalisation(s9ti);
s10ti=csvread('S10-TI.csv');
s10ti=normalisation(s10ti);

TI = [s1ti,s2ti,s4ti,s5ti,s6ti,s8ti,s9ti,s10ti];
[TI_train,TI_test,TI_train_raw,TI_test_raw,TI_train_raw_target,...
 TI_test_raw_target,TI_train_split,TI_test_split,TI_train_split_target,...
 TI_test_split_target] = EMGdataprocess_group_2_2(TI,2);

%% TT class(output=10)
s1tt=csvread('S1-TT.csv');
s1tt=normalisation(s1tt);
s2tt=csvread('S2-TT.csv');
s2tt=normalisation(s2tt);
%s3tt=csvread('S3-TT.csv');
%s3ttft=extractfeatures(normalisation(s3tt));
s4tt=csvread('S4-TT.csv');
s4tt=normalisation(s4tt);
s5tt=csvread('S5-TT.csv');
s5tt=normalisation(s5tt);
s6tt=csvread('S6-TT.csv');
s6tt=normalisation(s6tt);
%s7tt=csvread('S7-TT.csv');
%s7ttft=extractfeatures(normalisation(s7tt));
s8tt=csvread('S8-TT.csv');
s8tt=normalisation(s8tt);
s9tt=csvread('S9-TT.csv');
s9tt=normalisation(s9tt);
s10tt=csvread('S10-TT.csv');
s10tt=normalisation(s10tt);

TT = [s1tt,s2tt,s4tt,s5tt,s6tt,s8tt,s9tt,s10tt];
[TT_train,TT_test,TT_train_raw,TT_test_raw,TT_train_raw_target,...
 TT_test_raw_target,TT_train_split,TT_test_split,TT_train_split_target,...
 TT_test_split_target] = EMGdataprocess_group_2_2(TT,3);


Target_raw_test = [M_test_raw_target TI_test_raw_target TT_test_raw_target];


Input_train_split_data = [M_train_split TI_train_split TT_train_split];

Target_train_split_data = [M_train_split_target TI_train_split_target TT_train_split_target];

Input_test_split_data = [M_test_split TI_test_split TT_test_split];

Target_test_split_data = [M_test_split_target TI_test_split_target TT_test_split_target];
