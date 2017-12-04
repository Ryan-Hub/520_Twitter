%%%TESTING
load("train.mat");
X = X_train_bag;
Y = Y_train;
% [X_most_freq] = mostusedwords(X_train_bag);
costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
K = 10;
err_vec = zeros(K,1);
for N = 1:K 
    [X_all, Y_all, X_test_all, Y_test_all,X_prop, Y_prop, X_test_prop, Y_test_prop] = CV(X, Y, K, N);
    alltreebag = TreeBagger(100,full(X_all), Y_all,'Cost', costs);
    save('alltreebag.mat', 'alltreebag');

    Y_hat = predict_labels(X_test_all, "SHIT");
    % Y_HAT = cell2mat(Y_hat);
    % Y_hat = str2num(Y_HAT);
    err = performance_measure(Y_hat, Y_test_all);
    disp(err);
    err_vec(N) = err;
end
avg_err = mean(err_vec);