N = 10;
X = X_train_bag;
Y = Y_train;


%  
joy= X(Y==1, :);
sadness = X(Y==2, :);
surprise = X(Y==3, :);
anger = X(Y==4, :);
fear = X(Y==5, :);

cv_index_all = crossvalind('KFold', 18092, 10)
cv_index_joy = crossvalind('KFold', 4389, 10)
cv_index_sadness = crossvalind('KFold', 6454, 10)
cv_index_surprise = crossvalind('KFold', 1570, 10)
cv_index_anger = crossvalind('KFold', 1933, 10)
cv_index_fear = crossvalind('KFold', 3746, 10)

