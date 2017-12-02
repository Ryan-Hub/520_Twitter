vocabdouble = full(X_train_bag); % Turns from sparse double to double matrix

% Sums the number of total occurance for each vocab word and links it to
% the vocab word in the vocabulary data set
for i = 1:10000
    
    vocabsum(i,1) = sum(vocabdouble(:,i)); 
    vocabsum(i,2) = i;
    
end

sorted = sortrows(vocabsum,1,'descend'); % Sorts data by vocab words that occur the most
%1:2177
% disp(sorted(1:2177,2))

% disp(X_train_bag(:,sorted(1:2177,2)))
test = sorted(1:100,2);
x_mostfreq = X_train_bag(:,sorted(1:2177,2));
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