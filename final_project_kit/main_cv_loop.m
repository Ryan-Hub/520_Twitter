%%%TESTING
clear;
addpath('./libsvm');
addpath('./liblinear');
addpath('./data');
addpath('./models');
addpath('./helper');
addpath('./cv_loop_scripts');
load("train.mat");
X = X_train_bag;
Y = Y_train;
% [X_most_freq] = mostusedwords(X_train_bag);
K = 10;
err_vec = zeros(K,1);
joy= X(Y==1, :);
sadness = X(Y==2, :);
surprise = X(Y==3, :);
anger = X(Y==4, :);
fear = X(Y==5, :);
cv_index_all = crossvalind('KFold', size(X,1), K);
cv_index_joy = crossvalind('KFold', size(joy,1), K);
cv_index_sadness = crossvalind('KFold', size(sadness,1), K);
cv_index_surprise = crossvalind('KFold', size(surprise,1), K);
cv_index_anger = crossvalind('KFold', size(anger,1), K);
cv_index_fear = crossvalind('KFold', size(fear,1), K);

disp('start CV');
best_weight = [0, 0];
best_weight_vec = zeros(K, 2);
for N = 1:K
    [~, ~, ~, ~, X_prop, Y_prop, X_test_prop, Y_test_prop] = CV(X, Y, N, cv_index_all, cv_index_joy, cv_index_sadness, cv_index_fear, cv_index_surprise, cv_index_anger);
    
%     old_do_stuff;
%     find_weights;
%     lin_reg;
%     combined_nb_lr;
%     try_method;
    active_learning;

    best_weight_vec(N, :) = best_weight;
    disp([' Fold-', num2str(N), ' performance: ', num2str(err)]);
    err_vec(N) = err;
end
avg_err = mean(err_vec);
disp(['avg performance: ', num2str(avg_err)]);
avg_weights = mean(best_weight_vec, 1);
disp(['avg_weights: ', num2str(avg_weights)]);
