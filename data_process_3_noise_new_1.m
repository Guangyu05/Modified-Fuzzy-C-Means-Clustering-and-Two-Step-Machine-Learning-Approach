
clear
% There are 8 subjects, you can specify the subjects for training and test
% respectively.

%%% Obtain input data.
tic
m = 0;
%rng('default')
%% HC class (output = 1)
s1hc=csvread('S1-HC.csv');
s2hc=csvread('S2-HC.csv');
s3hc=csvread('S3-HC.csv');
s4hc=csvread('S4-HC.csv');
s5hc=csvread('S5-HC.csv');
s6hc=csvread('S6-HC.csv');
s7hc=csvread('S7-HC.csv');
s8hc=csvread('S8-HC.csv');

HC = [s1hc s2hc s3hc s4hc s5hc s6hc s7hc s8hc];
HC1 = HC + m*randn(size(HC)); %If m = 0, then no noise added.
HC = newnorm(HC1);

[HC_test_raw_target,HC_train_split,HC_test_split,HC_train_split_target,...
 HC_test_split_target] = EMGdataprocess3_new(HC,1);
%% I class(output=2)
s1i=csvread('S1-I.csv');
s2i=csvread('S2-I.csv');
s3i=csvread('S3-I.csv');
s4i=csvread('S4-I.csv');
s5i=csvread('S5-I.csv');
s6i=csvread('S6-I.csv');
s7i=csvread('S7-I.csv');
s8i=csvread('S8-I.csv');

I = [s1i,s2i,s3i,s4i,s5i,s6i,s7i,s8i];
I1 = I + m*randn(size(I));
I = newnorm(I1);

[I_test_raw_target,I_train_split,I_test_split,I_train_split_target,...
 I_test_split_target] = EMGdataprocess3_new(I,2);

%% L class(output=3)
s1l=csvread('S1-L.csv');
s2l=csvread('S2-L.csv');
s3l=csvread('S3-L.csv');
s4l=csvread('S4-L.csv');
s5l=csvread('S5-L.csv');
s6l=csvread('S6-L.csv');
s7l=csvread('S7-L.csv');
s8l=csvread('S8-L.csv');

L = [s1l,s2l,s3l,s4l,s5l,s6l,s7l,s8l];
L1 = L + m*randn(size(L));
L = newnorm(L1);

[L_test_raw_target,L_train_split,L_test_split,L_train_split_target,...
 L_test_split_target] = EMGdataprocess3_new(L,3);

%% M class(output=4)
s1m=csvread('S1-M.csv');
s2m=csvread('S2-M.csv');
s3m=csvread('S3-M.csv');
s4m=csvread('S4-M.csv');
s5m=csvread('S5-M.csv');
s6m=csvread('S6-M.csv');
s7m=csvread('S7-M.csv');
s8m=csvread('S8-M.csv');

M = [s1m,s2m,s3m,s4m,s5m,s6m,s7m,s8m];
M1 = M + m*randn(size(M));
M = newnorm(M1);

[M_test_raw_target,M_train_split,M_test_split,M_train_split_target,...
 M_test_split_target] = EMGdataprocess3_new(M,4);

%% R class(output=5)
s1r=csvread('S1-R.csv');
s2r=csvread('S2-R.csv');
s3r=csvread('S3-R.csv');
s4r=csvread('S4-R.csv');
s5r=csvread('S5-R.csv');
s6r=csvread('S6-R.csv');
s7r=csvread('S7-R.csv');
s8r=csvread('S8-R.csv');

R = [s1r,s2r,s3r,s4r,s5r,s6r,s7r,s8r];
R1 = R + m*randn(size(R));
R = newnorm(R1);

[R_test_raw_target,R_train_split,R_test_split,R_train_split_target,...
 R_test_split_target] = EMGdataprocess3_new(R,5);

%% TI class(output=6)
s1ti=csvread('S1-TI.csv');
s2ti=csvread('S2-TI.csv');
s3ti=csvread('S3-TI.csv');
s4ti=csvread('S4-TI.csv');
s5ti=csvread('S5-TI.csv');
s6ti=csvread('S6-TI.csv');
s7ti=csvread('S7-TI.csv');
s8ti=csvread('S8-TI.csv');

TI = [s1ti,s2ti,s3ti,s4ti,s5ti,s6ti,s7ti,s8ti];
TI1 = TI + m*randn(size(TI));
TI = newnorm(TI1);

[TI_test_raw_target,TI_train_split,TI_test_split,TI_train_split_target,...
 TI_test_split_target] = EMGdataprocess3_new(TI,6);

%% TL class(output=7)
s1tl=csvread('S1-TL.csv');
s2tl=csvread('S2-TL.csv');
s3tl=csvread('S3-TL.csv');
s4tl=csvread('S4-TL.csv');
s5tl=csvread('S5-TL.csv');
s6tl=csvread('S6-TL.csv');
s7tl=csvread('S7-TL.csv');
s8tl=csvread('S8-TL.csv');

TL = [s1tl,s2tl,s3tl,s4tl,s5tl,s6tl,s7tl,s8tl];
TL1 = TL + m*randn(size(TL));
TL = newnorm(TL1);

[TL_test_raw_target,TL_train_split,TL_test_split,TL_train_split_target,...
 TL_test_split_target] = EMGdataprocess3_new(TL,7);

%% TM class(output=8)
s1tm=csvread('S1-TM.csv');
s2tm=csvread('S2-TM.csv');
s3tm=csvread('S3-TM.csv');
s4tm=csvread('S4-TM.csv');
s5tm=csvread('S5-TM.csv');
s6tm=csvread('S6-TM.csv');
s7tm=csvread('S7-TM.csv');
s8tm=csvread('S8-TM.csv');

TM = [s1tm,s2tm,s3tm,s4tm,s5tm,s6tm,s7tm,s8tm];
TM1 = TM + m*randn(size(TM));
TM = newnorm(TM1);

[TM_test_raw_target,TM_train_split,TM_test_split,TM_train_split_target,...
 TM_test_split_target] = EMGdataprocess3_new(TM,8);

%% TR class(output=9)
s1tr=csvread('S1-TR.csv');
s2tr=csvread('S2-TR.csv');
s3tr=csvread('S3-TR.csv');
s4tr=csvread('S4-TR.csv');
s5tr=csvread('S5-TR.csv');
s6tr=csvread('S6-TR.csv');
s7tr=csvread('S7-TR.csv');
s8tr=csvread('S8-TR.csv');

TR = [s1tr,s2tr,s3tr,s4tr,s5tr,s6tr,s7tr,s8tr];
TR1 = TR + m*randn(size(TR));
TR = newnorm(TR1);

[TR_test_raw_target,TR_train_split,TR_test_split,TR_train_split_target,...
 TR_test_split_target] = EMGdataprocess3_new(TR,9);

%% TT class(output=10)
s1tt=csvread('S1-TT.csv');
s2tt=csvread('S2-TT.csv');
s3tt=csvread('S3-TT.csv');
s4tt=csvread('S4-TT.csv');
s5tt=csvread('S5-TT.csv');
s6tt=csvread('S6-TT.csv');
s7tt=csvread('S7-TT.csv');
s8tt=csvread('S8-TT.csv');

TT = [s1tt,s2tt,s3tt,s4tt,s5tt,s6tt,s7tt,s8tt];
TT1 = TT + m*randn(size(TT));
TT = newnorm(TT1);

[TT_test_raw_target,TT_train_split,TT_test_split,TT_train_split_target,...
 TT_test_split_target] = EMGdataprocess3_new(TT,10);

%%
Target_test_raw_data = [HC_test_raw_target I_test_raw_target L_test_raw_target M_test_raw_target ...
R_test_raw_target TI_test_raw_target TL_test_raw_target TM_test_raw_target TR_test_raw_target TT_test_raw_target];

Input_train_split_data = [HC_train_split I_train_split L_train_split M_train_split ...
R_train_split TI_train_split TL_train_split TM_train_split TR_train_split TT_train_split];

Target_train_split_data = [HC_train_split_target I_train_split_target L_train_split_target M_train_split_target ...
R_train_split_target TI_train_split_target TL_train_split_target TM_train_split_target TR_train_split_target TT_train_split_target];

Input_test_split_data = [HC_test_split I_test_split L_test_split M_test_split ...
R_test_split TI_test_split TL_test_split TM_test_split TR_test_split TT_test_split];

Target_test_split_data = [HC_test_split_target I_test_split_target L_test_split_target M_test_split_target ...
R_test_split_target TI_test_split_target TL_test_split_target TM_test_split_target TR_test_split_target TT_test_split_target];
toc
