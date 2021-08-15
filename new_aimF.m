function obj_Fcn = new_aimF(data, cluster_n, class_n, U, C, m)

b = 0;

for i = 1:cluster_n
    for j = 1:class_n
        a = 0;
        for k = 1:size(data,1)/class_n
            a = a + norm(data((j-1)*32+k,:) - C(i,:))^2;
        end
        b = b + (U(i,j)^m)*a;
    end
end
obj_Fcn=b;