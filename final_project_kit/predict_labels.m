function [Y_hat] = predict_labels(X_test_bag, test_raw)


% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
    %% Naive Bayes on bag of words
    
    load('model_nb.mat', 'model_nb');
    [~ ,prob_estimates_nb, ~] = predict(model_nb, X_test_bag);
    % Y_hat = probability_to_class(prob_estimates_nb);
    
    %% length of tweets + multinomial logistic regression
    
    
    %% liblinear logreg
    addpath('./liblinear');
    load('model_logreg.mat', 'model_logreg');

    % model_logreg = train(Y_train, X_train_bag, '-s 0');
    [~, ~, prob_estimates_logreg] = predict(zeros(size(X_test_bag, 1), 1), X_test_bag, model_logreg, '-b 1');

    prob_estimates_logreg(:,[2,5]) = prob_estimates_logreg(:,[5,2]);
    % Y_hat = probability_to_class(prob_estimates);
    
    
    %% ensamble
    prob_estimates_total = prob_estimates_logreg + prob_estimates_nb;
    
    %% probability to class
    costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    [~, Y_hat] = min(prob_estimates_total*costs, [], 2);
     
end