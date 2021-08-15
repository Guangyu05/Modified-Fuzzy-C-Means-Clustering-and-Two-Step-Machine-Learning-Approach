%clear

load NN_input_test.mat
load NN_target_test_10.mat
load NN_input_train.mat
load NN_target_train_10.mat


target_train = zeros(size(Train_Input,1),1);

target_train([1:32,65:96,129:160,289:320],:) = 1; % class 1,3,5,10

target_train([97:128,161:192,193:224],:) = 2; %class 4,6,7

target_train([33:64,225:256,257:288],:) = 3; %class 2,8,9

target_test = zeros(size(Test_Input,1),1);

target_test([1:16,33:48,65:80,145:160],:) = 1; % class 1,3,5,10

target_test([49:64,81:96,97:112],:) = 2; % class 4,6,7

target_test([17:32,113:128,129:144],:) = 3; %class 2,8,9

Train_output_3 = zeros(3,size(Train_Output,2));
for i = 1:size(Train_Output,2)
    Train_output_3(target_train(i),i) = 1;
end

Test_output_3 = zeros(3,size(Test_Output,2));
for i = 1:size(Test_Output,2)
    Test_output_3(target_test(i),i) = 1;
end

x = Train_Input;
t = Train_output_3;
rng('default')
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm'; % Levenberg-Marquardt backpropagation.
% Create a Fitting Network
% Configure the number of layers and neurons of neural network.
hiddenLayerSize = [25 25 25];
net = fitnet(hiddenLayerSize,trainFcn);
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand'; % Divide data randomly
net.divideMode = 'sample'; % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse'; % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
'plotregression', 'plotfit'};
% Train the Network
net.trainParam.epochs=500;
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);
% After training, the weight matrices of layers were saved and ready for
% reusing in VWNN. Once the weights are saved in .mat files, VWNN can
% directly use data from external files instead of internal variables in
% traditional neural network.
%Weight_1st_layer=net.iw{1,1};
%save Weight_1st_layer Weight_1st_layer
%Weight_2nd_layer=net.lw{2,1};
%save Weight_2nd_layer Weight_2nd_layer
% View the Network
view(net);
% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
% Generate MATLAB function for neural network for application
% deployment in MATLAB scripts or with MATLAB Compiler and Builder
% tools, or simply to examine the calculations your trained neural
% network performs.
genFunction(net,'myNeuralNetworkFunction');
y = myNeuralNetworkFunction(x);
end
if (false)
% Generate a matrix-only MATLAB function for neural network code
% generation with MATLAB Coder tools.
genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
y = myNeuralNetworkFunction(x);
end
if (false)
% Generate a Simulink diagram for simulation or deployment with.
% Simulink Coder tools.
gensim(net);

end
% Calculate the training and testing accuracy and plot the confusion
% matrices.

load top_net_best.mat

y_train = net(Train_Input);
y_test = net(Test_Input);
temp = max(y_test);
y_test_label = zeros(size(y_test,2),1);
for i = 1:size(y_test,2)
    temp1 = find(y_test(:,i) == temp(i));
    y_test_label(i) = temp1;
end

testingaccuracy = size(find(target_test == y_test_label),1)/size(y_test,2)

target_train = zeros(size(Train_Input,1),1);

target_train([1:32,129:160,193:224,257:288],:) = 1; % class 1,5,7,9

target_train([33:64,65:96,225:256]) = 2;  %class 2,3,8

target_train([97:128,161:192,289:320],:) = 3; %class 4,6,10

target_test = zeros(size(Test_Input,1),1);

target_test([1:16,65:80,97:112,129:144],:) = 1; % class 1,5,7,9

target_test([17:32,33:48,113:128],:) = 2; %class 2,3,8

target_test([49:64,81:96,145:160],:) = 3; %class 4,6,10

% trainingaccuracy=trainingresult(net,Train_Input,Train_output_3)
% testingaccuracy1=testresult(net,Test_Input,Test_output_3)

load Input_test_raw_data.mat

final_predict_label = zeros(size(y_test_label));

for i = 1:size(Test_Input,2)
    indices = y_test_label(i);
    temp_data = Input_test_raw_data{i};
    temp_data = testsplit(temp_data);
    switch indices
        case 1
        temp_label = double(mode(classify(cnn_goup_1,temp_data)));
        switch temp_label
            case 1
            final_predict_label(i) = 1;
            case 2
            final_predict_label(i) = 5;
            case 3
            final_predict_label(i) = 7;    
            case 4
            final_predict_label(i) = 9;   
        end
        case 2
        temp_label = double(mode(classify(cnn_goup_2,temp_data)));
        switch temp_label
            case 1
            final_predict_label(i) = 2;
            case 2
            final_predict_label(i) = 3;
            case 3
            final_predict_label(i) = 8;    
        end    
        case 3
        temp_label = double(mode(classify(cnn_goup_3,temp_data)));
        switch temp_label
            case 1
            final_predict_label(i) = 4;
            case 2
            final_predict_label(i) = 6;
            case 3
            final_predict_label(i) = 10;    
        end 
    end
end

test_target = zeros(160,1);
for i = 1:size(test_target,1)
test_target(i) = find(Test_Output(:,i) == 1);
end
            

final_accuracy = size(find(test_target == final_predict_label),1)/160
        
        
    

%% KNN
% a = Train_Input';
% 
% b = Test_Input';
% 
% Mdl = fitcknn(a,target_train,'NSMethod','exhaustive','NumNeighbors',3,'Distance','cityblock');
% 
% Predict_Train = predict(Mdl,a);
% 
% predict_test = predict(Mdl,b);
% 
% temp = find(predict_test ~= target_test);
% 
% test_accuracy = 1-size(temp,1)/size(target_test,1)
%% SVM
a = Train_Input';

b = Test_Input';

t = templateSVM('Standardize',1);

rng default

Mdl = fitcecoc(a,target_train,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))

CVMdl = crossval(Mdl)

loss = kfoldLoss(CVMdl)


