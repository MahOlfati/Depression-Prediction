% ----------------------------------------------------------------------- %
%                      S L E E P     P R E D I C T O R                    %
% ----------------------------------------------------------------------- %
% This script predicts depressive symptoms of Achenbach questionnaire     %
% based on the 20 features of Pittsburgh sleep quality index (PSQI).      %
%                                                                         %
%   Input parameters:                                                     %
%       - sleep:        Sleep quality parameters.                         %
%       - depression:   Achenbach questionnaire scores.                   %
%       - family:       Family IDs for considering family structure to    %
%                       avoid siblings to be seperated in training/test   %
%                       sets.                                             %
%       - confounding:  Age,gender, ant tGMV considered as confound       %
%                       variables.                                        %
%       - anxi:         Anxiety scores.                                   %
%                                                                         %
%   Output variables:                                                     %
%       - newmodel:     Models that were creared by machine learning      %
%                       algorithm for individual prediction.              %
%       - YHat:         Predicted values.                                 %
%       - ls1:          Loss-function of prediction in test sets based on %
%                       minimum mean squared error (MSE).                 %
%       - R2:           Coefficient of determination.                     %
%       - perf:         Mean absolute error.                              %
%       - ci:           95% confidence interval.                          %
%       - MSE:          Mean squared error.                               %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:      1.2.                                              %
%       - Author:       Mahnaz Olfati                                     %
%       - Date:         21/02/2022                                        %
% ----------------------------------------------------------------------- %
% Read data
clc,clear,close all




sleep                  % Put sleep variable here
depression             % Put depression varible here 
confounding            % Put confounding variables here
family                 % Put family ID here
anxi                   % Put anxiety variable here

%% Predictor and target vriables
x = sleep;             % Put predictors here
y = depression;        % Put target here

%% Nested 10-fold cross-validation considering the family structure
kfold_o = 10;          % Number of outer folds
kfold_i = 10;          % NUmber of inner folds
[test_idx,train_outer_idx,train_inner_idx,validation_idx] = NestedCV(y,kfold_o,kfold_i,family);

%% Training models, saving the best models, and fitting selected models on outer folds

feature_num = size(x,2);
conf_mdl = cell(kfold_o,feature_num);

for outer = 1:kfold_o      % Outer folds

    y_test = y(test_idx{outer},1); 
    y_train = y(train_outer_idx{outer},1);
    x_test = x(test_idx{outer},:);
    x_train = x(train_outer_idx{outer},:);
    conftrain_outer = confounding(train_outer_idx{outer},:);
    conftest = confounding(test_idx{outer},:);
   
    % creation of confound removal models (conf_mdl) on training sets and applying them on test sets
    [x_train,x_test,conf_mdl] = Confound_Remove_model(outer,x_train,x_test,conftrain_outer,conftest,conf_mdl);
    
    % Ranking features in outer training folds
    rng default
    [ranks{outer},weights{outer}] = relieff(x_train,y_train,10);

    % Training models for each inner fold
    parfor inner = 1:kfold_i    % Inner folds

        x_trainn = x(train_inner_idx{outer,inner},ranks{outer}(1:feature_num));
        x_validation = x(validation_idx{outer,inner},ranks{outer}(1:feature_num));
        y_trainn = y(train_inner_idx{outer,inner},1);
        y_validation = y(validation_idx{outer,inner},1);
        conftrain_inner = confounding(train_inner_idx{outer,inner},:);
        conf_validation = confounding(validation_idx{outer,inner},:);
        
        % regressing out the cnfound variables from inner training and validation predictors
        [x_trainn,x_validation] = RegressOut(outer,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num);
        
        % Training models on inner training folds
        Mdl1{inner}= Model(x_trainn,y_trainn)
        
        % Assessment of loss-function of validation sets to select model with best performance
        ls{outer,inner} = loss(Mdl1{inner},x_validation,y_validation);
    end

    % Selection of best model for each outer fold based on minimum loss of ineer folds
    [lost{outer},best{outer}] = min(cell2mat(ls(outer,:)));
    bestmodel{outer} = Mdl1{best{outer}};
    
    % fitting selected model on the entire training sets and prediction of test sets
    [newmodel{outer}, ls1{outer},Yhat1{outer}] = fitmodel(bestmodel{outer},x_train,y_train,x_test,y_test,ranks{outer});
     
end

%% Predicted values
YHat(cell2mat(test_idx')) = cell2mat(Yhat1');
YHat = YHat';

%% Plotting
reg = [YHat,y];
figure;
[R,PValue] = corrplot(reg,'testR','on')

R2 = 1 - sum((y - YHat) .^ 2) / sum((y - mean(y)) .^ 2);

e = YHat-y;
perf = mae(e);

% 95% confidence interval
ts = tinv([0.025 0.975],length(YHat)-1);
sem = std(YHat)/sqrt(length(YHat));
ci = mean(YHat)+ts*sem

MSE = mean(cell2mat(ls1))

%% Storage of models for out of sample validation
save('modelanxi_sleep_conf.mat','newmodel','ranks','conf_mdl')
