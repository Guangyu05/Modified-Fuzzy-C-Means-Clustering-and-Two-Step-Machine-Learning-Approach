function final_xTest = testsplit(y)

Window = 500;
Incre = 10;
Datalength = 20000;

split_samples = (Datalength - Window)/Incre + 1;

y_test_split_new = {};

for j = 0:split_samples-1
    y_test_split_new{j+1} = y(Incre*j+1:Incre*j+Window,:);
end

final_xTest = zeros(Window,2,1,numel(y_test_split_new));
a = cell2mat(y_test_split_new); 
b = zeros(Window,2,1,numel(y_test_split_new));
for i = 1:numel(y_test_split_new)
    b(:,:,1,i) = a(:,((i-1)*2+1):i*2);
end
final_xTest = b;

