features_filename = fullfile('results','reviewer_classification','toy_features.csv');
labels_filename = fullfile('results','reviewer_classification','toy_labels.csv');
k = 10; % number of neighbors for reliefF
nFeatures = [50, 70, 100, 30, 15, 7]; % number of features for each feature type
feature_names = {'unigram hist',...
                 'bigram hist',...
                 'trigram hist',...
                 'fourgram hist',...
                 'fivegram hist',...
                 'LM perplexity'};

X = csvread(features_filename);
y = csvread(labels_filename);

% X = [-1,2,3,4,5,3,6,2,7,3,6;...
%       -1,5,2,6,-1,3,5,2,5,-8,3;...
%       -1,2,-2,43,1,4,4,4,-3,-1,-1;...
%       -10,2,2,2,2,0,0,0,-3,2,2;...
%       1,2,0,-1,-1,-1,-1,-2,-3,-4,-5;...
%       1,2,1,2,3,4,5,6,-3,8,9;...
%       1,0,0,0,0,0,0,0,0,0,0];
% y = [1,1,1,1,0,0,0].';

[idx,weights] = relieff(X,y,k);

figure();
hold on;
ind = 1;
for i = 1:length(nFeatures)
    bar(ind:(ind+nFeatures(i)-1), weights(ind:(ind+nFeatures(i)-1)))
    ind = ind + nFeatures(i);
end
grid on;
title('Feature Importance');
xlabel('Feature');
ylabel('Feature importance weight');
legend(feature_names);
