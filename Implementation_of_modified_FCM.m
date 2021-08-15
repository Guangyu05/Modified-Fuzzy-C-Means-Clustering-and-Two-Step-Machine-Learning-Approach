load a1_for_fcm.mat
load a2_for_fcm.mat

[C, U, obj_Fcn] = new_fcm(a, 6, 10, 100, 50, 0.0001);
[max_value, index1] = max(U)
[C, U, obj_Fcn] = new_fcm(a1, 6, 10, 100, 50, 0.0001)
[max_value, index2] = max(U)
