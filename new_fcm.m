function [C, U, obj_Fcn] = new_fcm(data, cluster_n, class_n, m, max_iter, min_impro)

obj_Fcn = zeros(max_iter, 1);	% Array for objective function

%U = new_initfcm(cluster_n, class_n, data, center, m);			% Initial fuzzy partition
U = rand(cluster_n,class_n);
U=U./repmat(sum(U),cluster_n,1);
%C = rand(3,4);
% Main loop
for i = 1:max_iter
    C = new_center(cluster_n, class_n, data, m, U);
    U = new_initfcm(cluster_n, class_n, data, C,m);
    i
    obj_Fcn(i)=new_aimF(data, cluster_n, class_n, U, C, m);
	if i > 1
		if abs(obj_Fcn(i) - obj_Fcn(i-1)) < min_impro, break; end
	end
end

iter_n = i;	% Actual number of iterations 
obj_Fcn(iter_n+1:max_iter) = [];

%[max_value, index] = max(U);

%[C, U, obj_Fcn] = new_fcm(a, 3, 10, 20, 50, 0.0001)
%[max_value, index] = max(U)
%[C, U, obj_Fcn] = new_fcm(a, 3, 10, 200, 20, 0.0001)

%[C, U, obj_Fcn] = new_fcm(a1, 3, 10, 50, 50, 1)


%[C, U, obj_Fcn] = new_fcm(a, 3, 10, 500, 50, 1)
%[max_value, index] = max(U)
%[C, U, obj_Fcn] = new_fcm(a, 10, 10, 10, 20, 1)


%[C, U, obj_Fcn] = new_fcm(a, 6, 10, 100, 50, 0.0001)
%[max_value, index] = max(U)
%1     8     6    10     1    10     1    10     1    10

%[C, U, obj_Fcn] = new_fcm(a1, 6, 10, 100, 50, 0.0001)
%[max_value, index] = max(U)
%5     4     5     4     5     4     5     1     5     4