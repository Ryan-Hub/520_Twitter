%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script: findpc
%Author: Ryan Hub
% 
%Purpose: Takes X_train_bag and converts it from
%a sparse double matrix to a double matrix. It then
%determines and sorts all vocab words from tweets and 
%ranks in descending order. It finally plots the amount 
%of words based off percentage of total appearance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('/Users/ryanhub/Desktop/Penn CIS/Fall 2017/CIS 520/Final Project/final_project_kit/train.mat')

vocabdouble = full(X_train_bag); % Turns from sparse double to double matrix

% Sums the number of total occurance for each vocab word and links it to
% the vocab word in the vocabulary data set
for i = 1:10000
    
    vocabsum(i,1) = sum(vocabdouble(:,i)); 
    vocabsum(i,2) = i;
    
end

sorted = sortrows(vocabsum,1,'descend'); % Sorts data by vocab words that occur the most


newsort = sorted(:,1); % Extracts just the number of occurances from the sorted data
figure, plot(cumsum(newsort)/sum(newsort)); % plots the amount of words based off percentage of total appearance

% Used to find the amount words neccesary to represent the total amount of
% words observed
for x = 1:10000
    h = (cumsum(newsort(1:x))/sum(newsort));
    if (h(x) >= 0.9)
        numpc = x;
        disp(numpc);
        break;
    end

end

