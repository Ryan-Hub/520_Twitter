   
    addpath('./liblinear');
  
    model_lr = train(Y_prop, double(logical(X_prop)), '-s 0 -q 1');
    [~, ~, prob_lr] = predict(Y_test_prop, double(logical(X_test_prop)), model_lr, '-b 1');
%     prob_lr(:,[2,5]) = prob_lr(:,[5,2]);
    
    Y_hat = probability_to_class(prob_lr);
    err = performance_measure(Y_hat, Y_test_prop);