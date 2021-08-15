# Modified-Fuzzy-C-Means-Clustering-and-Two-Step-Machine-Learning-Approach
This is the source code for the paper 'Classification of Electromyographic Hand Gesture Signals Using Modified Fuzzy C-Means Clustering and Two-Step Machine Learning Approach (DOI:10.1109/TNSRE.2020.2986884)'

### Dataset
The data used in this paper is from R. N. Khushaba, M. Takruri, S. Kodagoda, and G. Dissanayake, "Toward Improved Control of Prosthetic Fingers Using Surface Electromyogram (EMG) Signals", Expert Systems with Applications, vol 39, no. 12, pp. 10731–10738, 2012,  and can be downloaded from https://www.rami-khushaba.com/electromyogram-emg-repository.html


### Unsupervised learning part:

####Find three groups

source code: 

	Implementation_of_modified_FCM.m

- Run the file Implementation\_of\_modified\_FCM.m, we can find from index1 and index2 that the class 2,3 and 8 are easy to be distinguished with other classes. Thus, we have the following division of the classes:

     1. Group one: classes 1,5,7,9
     2. Group two: classes 2,3,8
     3. Group three: classes 4,6,10

### Supervised learning part:

####Train the top model

source code: 
	
	NN_top_group_3.m

- Use this file to train the top NN model. The number of nodes for the hidden layers is [25, 25, 25].The trained model (accuracy:98.75%) is saved in the file:

		top_net_best.mat

##### Other classifiers:
source code: 

	NN_KNN_SVM_RF_DF.py
	run rf top classifier  accuracy:96.25%, 0.996875
	run knn-3  accuracy:95%, 0.9875
	run dt-top-3 accuracy: 0.9313, 1
After windowing:

	In matlab:
	run Process_10_classes_to_3_classes_14_01_2020.m
	run Feature_split
	save related data

	NN.py ####For the top classifier, data:19/01/2020
	
	rf training accuracy:0.9956
	   test accuracy(after voting): 0.9625
	dt training accuracy:1
	   test accuracy(after voting): 0.975
	knn training accuracy: 0.9116
		test accuracy(after voting): 0.925

	
	 

#### Obtain the windowing data of each group

source code: 

    1.new_group1.m
	2.new_group2.m
	3.new_group3.m

- Use these three files to create the windowing data in each group.

#### Do the classification in each group


source code: 

    1.CNN_group_1_1.m 
	2.CNN_group_2_1.m
	3.CNN_group_2_2.m

- Use the above three files to train the individual models. The training time of each is no more than 10min. The saved trained CNN is names as:
 

		cnn, cnn_goup_2, cnn_goup_3

The above models are saved respectively in files:

	cnn_group_1_new.mat, cnn_group_2_new.mat, cnn_group_3_new.mat





#### Calculate the final classification accuracy


source code: 

	final_results_calculation.m

- Use this file to calculate the final accuracy. Actually, it is the multiplication of the top model's accuracy and the individual model's accuracy. The calculation should begin from the 117th line:
 

		load top_net_best.mat


The final accuracy is 98.75%, which is also the accuracy of the top classifier. 

###If we change the top model to CNN with windowing and majority voting, the results are as follows:

Top model accuracy: 100%

Overall accuracy: 100%

source code:

	Process_10_classes_to_3_classes_14_01_2020.m
	CNN_top_for_3_classes.m

	% accuracy_train_cnn =
	% 
	%     0.9970
	% 
	% 
	% accuracy_test_cnn =
	% 
	%     0.9498
	% 
	% 
	% accuracy_after_voting =
	% 
	%      1
 

The best top model is saved as cnn_best_top_100_new.mat and can only by loaded using Matlab version>2018b


