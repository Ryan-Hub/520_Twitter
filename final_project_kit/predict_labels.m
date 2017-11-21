function [Y_hat] = predict_labels(X_test_bag, test_raw)


% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
    %% Naive Bayes on bag of words
    
    load('model_nb.mat', 'model_nb');
    [~ ,Posterior, ~] = predict(model_nb, X_test_bag);
    Y_hat = probability_to_class(Posterior);
    
    %% length of tweets + multinomial logistic regression
    
    
    
end
