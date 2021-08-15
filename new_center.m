function C = new_center(cluster_n, class_n, data, m, U)

%C = zeros(cluster_n,size(data,2));

for i = 1:cluster_n 
    b = zeros(1,size(data,2));
    c = 0;
    for j = 1:class_n
            a = sum(data((j-1)*32+1:j*32,:));
            b = b + (U(i,j)^m)*a;
            c = c + U(i,j)^m;
    end 
    C(i,:) = b/(c*32);
end

