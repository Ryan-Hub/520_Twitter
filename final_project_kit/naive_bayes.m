% naive_bayes

%% load data
% load validation.mat
clear
load train.mat
% load vocabulary.mat


%% 
load fisheriris

X = full(X_train_bag);
X = X(1:150, :);
Y = species;
% Y = Y_train(1:150, :);
nb = fitcnb((X+1)*5, Y);

%%
load fisheriris
X = meas(:,3:4);
Y = species;
Mdl = fitcnb(X,Y);
%%