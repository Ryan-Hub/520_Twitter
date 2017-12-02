N = 10;


%  


function [X_all, Y_all, X_test_all, Y_test_all,X_prop, Y_prop, X_test_prop, Y_test_prop] = generate_CV(X, Y)
    joy= X(Y==1, :);
    sadness = X(Y==2, :);
    surprise = X(Y==3, :);
    anger = X(Y==4, :);
    fear = X(Y==5, :);
    cv_index_all = crossvalind('KFold', size(X,1), 10);
    cv_index_joy = crossvalind('KFold', size(joy,1), 10);
    cv_index_sadness = crossvalind('KFold', size(sadness,1), 10);
    cv_index_surprise = crossvalind('KFold', size(surprise,1), 10);
    cv_index_anger = crossvalind('KFold', size(anger,1), 10);
    cv_index_fear = crossvalind('KFold', size(fear,1), 10);

    X_all = X(cv_index_all ~= 10, :);
    Y_all = Y(cv_index_all ~= 10, :);

    X_test_all =  X(cv_index_all == 10, :);
    Y_test_all = Y(cv_index_all == 10, :);

    X_J = joy(cv_index_joy ~= 10, :);
    Y_J = ones(size(X_J,1),1);

    X_test_J =  joy(cv_index_joy == 10, :);
    Y_test_J = ones(size(X_test_J,1),1);

    X_S = sadness(cv_index_sadness ~= 10, :);
    Y_S(size(X_S,1),1) = 2;

    X_test_S = sadness(cv_index_sadness == 10, :);
    Y_test_S(size(X_test_S,1),1) = 2;

    X_SUR = surprise(cv_index_surprise ~= 10, :);
    Y_SUR(size(X_SUR,1),1) = 3;

    X_test_SUR = surprise(cv_index_surprise == 10, :);
    Y_test_SUR(size(X_test_SUR,1),1) = 3;

    X_A = anger(cv_index_anger ~= 10, :);
    Y_A(size(X_A,1),1) = 4;

    X_test_A = anger(cv_index_anger == 10, :);
    Y_test_A(size(X_test_A,1),1) = 4;

    X_F = fear(cv_index_fear ~= 10, :);
    Y_F(size(X_F,1),1) = 5;

    X_test_F = fear(cv_index_fear == 10, :);
    Y_test_F(size(X_test_F,1),1) = 5;

    X_prop = [X_J; X_S; X_SUR; X_A; X_F];
    Y_prop = [Y_J; Y_S; Y_SUR; Y_A; Y_F];

    X_test_prop =  [X_test_J; X_test_S; X_test_SUR; X_test_A; X_test_F];
    Y_test_prop = [Y_test_J; Y_test_S; Y_test_SUR; Y_test_A; Y_test_F];
end



