
%clear
% There are 8 subjects, you can specify the subjects for training and test
% respectively.
tic
%%% Obtain input data.
m = 0
%rng('default')
%% HC class (output = 1)
s1hc=csvread('S1-HC.csv');
s2hc=csvread('S2-HC.csv');
s3hc=csvread('s3-HC.csv');
s4hc=csvread('s4-HC.csv');
s5hc=csvread('s5-HC.csv');
s6hc=csvread('s6-HC.csv');
s7hc=csvread('s7-HC.csv');
s8hc=csvread('s8-HC.csv');

HC = [s1hc s2hc s3hc s4hc s5hc s6hc s7hc s8hc];
HC1 = HC + m*randn(size(HC)); %If m = 0, then no noise added.
HC = newnorm(HC1);

[HC_train,HC_test,HC_train_raw,HC_test_raw,HC_train_raw_target,...
 HC_test_raw_target,HC_train_split,HC_test_split,HC_train_split_target,...
 HC_test_split_target] = EMGdataprocess3(HC,1);
%% I class(output=2)
s1i=csvread('S1-I.csv');
s2i=csvread('S2-I.csv');
s3i=csvread('s3-I.csv');
s4i=csvread('s4-I.csv');
s5i=csvread('s5-I.csv');
s6i=csvread('s6-I.csv');
s7i=csvread('s7-I.csv');
s8i=csvread('s8-I.csv');

I = [s1i,s2i,s3i,s4i,s5i,s6i,s7i,s8i];
I1 = I + m*randn(size(I));
I = newnorm(I1);

[I_train,I_test,I_train_raw,I_test_raw,I_train_raw_target,...
 I_test_raw_target,I_train_split,I_test_split,I_train_split_target,...
 I_test_split_target] = EMGdataprocess3(I,2);

%% L class(output=3)
s1l=csvread('S1-L.csv');
s2l=csvread('S2-L.csv');
s3l=csvread('s3-L.csv');
s4l=csvread('s4-L.csv');
s5l=csvread('s5-L.csv');
s6l=csvread('s6-L.csv');
s7l=csvread('s7-L.csv');
s8l=csvread('s8-L.csv');

L = [s1l,s2l,s3l,s4l,s5l,s6l,s7l,s8l];
L1 = L + m*randn(size(L));
L = newnorm(L1);

[L_train,L_test,L_train_raw,L_test_raw,L_train_raw_target,...
 L_test_raw_target,L_train_split,L_test_split,L_train_split_target,...
 L_test_split_target] = EMGdataprocess3(L,3);

%% M class(output=4)
s1m=csvread('S1-M.csv');
s2m=csvread('S2-M.csv');
s3m=csvread('s3-M.csv');
s4m=csvread('s4-M.csv');
s5m=csvread('s5-M.csv');
s6m=csvread('s6-M.csv');
s7m=csvread('s7-M.csv');
s8m=csvread('s8-M.csv');

M = [s1m,s2m,s3m,s4m,s5m,s6m,s7m,s8m];
M1 = M + m*randn(size(M));
M = newnorm(M1);

[M_train,M_test,M_train_raw,M_test_raw,M_train_raw_target,...
 M_test_raw_target,M_train_split,M_test_split,M_train_split_target,...
 M_test_split_target] = EMGdataprocess3(M,4);

%% R class(output=5)
s1r=csvread('S1-R.csv');
s2r=csvread('S2-R.csv');
s3r=csvread('s3-R.csv');
s4r=csvread('s4-R.csv');
s5r=csvread('s5-R.csv');
s6r=csvread('s6-R.csv');
s7r=csvread('s7-R.csv');
s8r=csvread('s8-R.csv');

R = [s1r,s2r,s3r,s4r,s5r,s6r,s7r,s8r];
R1 = R + m*randn(size(R));
R = newnorm(R1);

[R_train,R_test,R_train_raw,R_test_raw,R_train_raw_target,...
 R_test_raw_target,R_train_split,R_test_split,R_train_split_target,...
 R_test_split_target] = EMGdataprocess3(R,5);

%% TI class(output=6)
s1ti=csvread('S1-TI.csv');
s2ti=csvread('S2-TI.csv');
s3ti=csvread('s3-TI.csv');
s4ti=csvread('s4-TI.csv');
s5ti=csvread('s5-TI.csv');
s6ti=csvread('s6-TI.csv');
s7ti=csvread('s7-TI.csv');
s8ti=csvread('s8-TI.csv');

TI = [s1ti,s2ti,s3ti,s4ti,s5ti,s6ti,s7ti,s8ti];
TI1 = TI + m*randn(size(TI));
TI = newnorm(TI1);

[TI_train,TI_test,TI_train_raw,TI_test_raw,TI_train_raw_target,...
 TI_test_raw_target,TI_train_split,TI_test_split,TI_train_split_target,...
 TI_test_split_target] = EMGdataprocess3(TI,6);

%% TL class(output=7)
s1tl=csvread('S1-TL.csv');
s2tl=csvread('S2-TL.csv');
s3tl=csvread('s3-TL.csv');
s4tl=csvread('s4-TL.csv');
s5tl=csvread('s5-TL.csv');
s6tl=csvread('s6-TL.csv');
s7tl=csvread('s7-TL.csv');
s8tl=csvread('s8-TL.csv');

TL = [s1tl,s2tl,s3tl,s4tl,s5tl,s6tl,s7tl,s8tl];
TL1 = TL + m*randn(size(TL));
TL = newnorm(TL1);

[TL_train,TL_test,TL_train_raw,TL_test_raw,TL_train_raw_target,...
 TL_test_raw_target,TL_train_split,TL_test_split,TL_train_split_target,...
 TL_test_split_target] = EMGdataprocess3(TL,7);

%% TM class(output=8)
s1tm=csvread('S1-TM.csv');
s2tm=csvread('S2-TM.csv');
s3tm=csvread('s3-TM.csv');
s4tm=csvread('s4-TM.csv');
s5tm=csvread('s5-TM.csv');
s6tm=csvread('s6-TM.csv');
s7tm=csvread('s7-TM.csv');
s8tm=csvread('s8-TM.csv');

TM = [s1tm,s2tm,s3tm,s4tm,s5tm,s6tm,s7tm,s8tm];
TM1 = TM + m*randn(size(TM));
TM = newnorm(TM1);

[TM_train,TM_test,TM_train_raw,TM_test_raw,TM_train_raw_target,...
 TM_test_raw_target,TM_train_split,TM_test_split,TM_train_split_target,...
 TM_test_split_target] = EMGdataprocess3(TM,8);

%% TR class(output=9)
s1tr=csvread('S1-TR.csv');
s2tr=csvread('S2-TR.csv');
s3tr=csvread('s3-TR.csv');
s4tr=csvread('s4-TR.csv');
s5tr=csvread('s5-TR.csv');
s6tr=csvread('s6-TR.csv');
s7tr=csvread('s7-TR.csv');
s8tr=csvread('s8-TR.csv');

TR = [s1tr,s2tr,s3tr,s4tr,s5tr,s6tr,s7tr,s8tr];
TR1 = TR + m*randn(size(TR));
TR = newnorm(TR1);

[TR_train,TR_test,TR_train_raw,TR_test_raw,TR_train_raw_target,...
 TR_test_raw_target,TR_train_split,TR_test_split,TR_train_split_target,...
 TR_test_split_target] = EMGdataprocess3(TR,9);

%% TT class(output=10)
s1tt=csvread('S1-TT.csv');
s2tt=csvread('S2-TT.csv');
s3tt=csvread('s3-TT.csv');
s4tt=csvread('s4-TT.csv');
s5tt=csvread('s5-TT.csv');
s6tt=csvread('s6-TT.csv');
s7tt=csvread('s7-TT.csv');
s8tt=csvread('s8-TT.csv');

TT = [s1tt,s2tt,s3tt,s4tt,s5tt,s6tt,s7tt,s8tt];
TT1 = TT + m*randn(size(TT));
TT = newnorm(TT1);

[TT_train,TT_test,TT_train_raw,TT_test_raw,TT_train_raw_target,...
 TT_test_raw_target,TT_train_split,TT_test_split,TT_train_split_target,...
 TT_test_split_target] = EMGdataprocess3(TT,10);

%%
Input_train = [HC_train I_train L_train M_train ...
R_train TI_train TL_train TM_train TR_train TT_train];


Input_train_raw_data = [HC_train_raw I_train_raw L_train_raw M_train_raw ...
R_train_raw TI_train_raw TL_train_raw TM_train_raw TR_train_raw TT_train_raw]; 

Target_train_raw_data = [HC_train_raw_target I_train_raw_target L_train_raw_target M_train_raw_target ...
R_train_raw_target TI_train_raw_target TL_train_raw_target TM_train_raw_target TR_train_raw_target TT_train_raw_target];

Input_test = [HC_test I_test L_test M_test ...
R_test TI_test TL_test TM_test TR_test TT_test];

Input_test_raw_data = [HC_test_raw I_test_raw L_test_raw M_test_raw ...
R_test_raw TI_test_raw TL_test_raw TM_test_raw TR_test_raw TT_test_raw];

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
