function [output] = newnorm(input)
for i = 1:size(input,2)/12
    temp1 = input(:,(i-1)*12+1:i*12);
    temp2 = rescale(temp1(:));
    output(:,(i-1)*12+1:i*12) = reshape(temp2,size(temp1));
end
end