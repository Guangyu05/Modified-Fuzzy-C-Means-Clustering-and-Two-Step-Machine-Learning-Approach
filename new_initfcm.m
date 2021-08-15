function U = new_initfcm(cluster_n, class_n, data, C, m)

data_n = size(data, 1);
%U = zeros(cluster_n,class_n);

for i = 1:cluster_n    
    for j = 1:class_n
        c = 0;
        for p = 1:cluster_n
            a = 0;
            b = 0;
            for k = 1:data_n/class_n
                a = a + norm(data((j-1)*32+k,:) - C(i,:))^2;
                b = b + norm(data((j-1)*32+k,:) - C(p,:))^2;
            end
            c = c + (a/b)^(1/(m-1));
        end
        U(i,j) = 1/c;
    end    
end

