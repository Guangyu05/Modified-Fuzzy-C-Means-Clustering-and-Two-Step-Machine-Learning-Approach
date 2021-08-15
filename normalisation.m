function[inputnorm]=normalisation(input)
m = 0;
input = input + m*randn(size(input));
FlattenedInput = input(:)';
MappedFlattenedInput = mapminmax(FlattenedInput, 0, 1); 
inputnorm = reshape(MappedFlattenedInput, size(input));
end

