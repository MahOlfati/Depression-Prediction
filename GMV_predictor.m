% ----------------------------------------------------------------------- %
%                      G M V     P R E D I C T O R                        %
% ----------------------------------------------------------------------- %
% This script predicts depressive symptoms of Achenbach questionnaire     %
% based on the 473 features of brains' gray matter volume (GMV) and       %
% combination of sleep quality and GMV features.                          %
%                                                                         %
%   Input parameters:                                                     %
%       - vol:          473 GMV features.                                 %
%       - sleep:        Sleep quality parameters.                         %
%       - depression:   Achenbach questionnaire scores.                   %
%       - family:       Family IDs for considering family structure to    %
%                       avoid siblings to be seperated in training/test   %
%                       sets.                                             %
%       - confounding:  Age,gender, ant tGMV considered as confound       %
%                       variables.                                        %
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
%       - Date:         25/12/2021                                        %
% ----------------------------------------------------------------------- %
% Read data
clc,clear,close all

sleep                  % Put sleep variable here
depression             % Put depression varible here 
confounding            % Put confounding variables here
family                 % Put family ID here
anxi                   % Put anxiety variable here
vol                    % Put GMV or fALFF or ReHo variable here

%% Predictor and target variables
x = vol;               % Put predictors here 
y = depression;        % Put target here

%% Nested 10-fold cross-validation considering the family structure
kfold_o = 10;          % Number of outer folds
kfold_i = 10;          % NUmber of inner folds
[test_idx,train_outer_idx,train_inner_idx,validation_idx] = NestedCV(y,kfold_o,kfold_i,family);

%% Training models on inner training folds, saving the best models, fitting selected models on the entire training sets and prediction of test sets

conf_mdl = cell(kfold_o,size(x,2));
for h = 1:kfold_o           % Outer folds

    y_test = y(test_idx{h},1); 
    y_train = y(train_outer_idx{h},1);
    x_test = x(test_idx{h},:);
    x_train = x(train_outer_idx{h},:);
    conftrain_outer = confounding(train_outer_idx{h},:);
    conftest = confounding(test_idx{h},:);
   
    % creation of confound removal models (conf_mdl) on training sets and applying them on test sets
        [x_train,x_test,conf_mdl] = Confound_Remove_model(h,x_train,x_test,conftrain_outer,conftest);

    
    % Ranking features in outer training folds
    rng default
    [ranks{h},weights{h}] = relieff(x_train,y_train,10);

    % Training models for each inner fold
    for i = 1:kfold_i       % Inner folds
        
        % Repeating training process for 10 different feature number
        parfor j = 1:10   
        
           feature_num{h,i,j} = 10+10*j;      % Number of features are included 20,30,40,50,60,70,80,90,100,110
            x_trainn = x(train_inner_idx{h,i},ranks{h}(1:feature_num{h,i,j}));
            x_validation = x(validation_idx{h,i},ranks{h}(1:feature_num{h,i,j}));
            y_trainn = y(train_inner_idx{h,i},1);
            y_validation = y(validation_idx{h,i},1);
            conftrain_inner = confounding(train_inner_idx{h,i},:);
            conf_validation = confounding(validation_idx{h,i},:);
            
            % regressing out the cnfound variables from inner training and validation predictors
            [x_trainn,x_validation] = RegressOut(h,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num{h,i,j})
            
            % Training models on inner training folds
            Mdl1{i,j}= Model(x_trainn,y_trainn)
            
            % Assessment of loss-function of validation sets to select optimum feature number and model with best performance
            ls{h,i,j} = loss(Mdl1{i,j},x_validation,y_validation);
        end
        
    end
    
    % Selection of best models for each outer fold based on minimum loss of ineer folds
    v = [ls{h,:,:}];
    [lost{h},best{h}] = min(v);
    [mj{h},mk{h}] = ind2sub([i,j],best{h});
    bestmodel{h} = Mdl1{mj{h},mk{h}}; 
    
    % Fitting selected models on the entire training sets and prediction of test sets
    rng default
    tmp{h} = classreg.learning.FitTemplate.makeFromModelParams(bestmodel{h}.ModelParameters);
    newmodel{h} = fit(tmp{h},x_train(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})),y_train);
    ls1{h} = loss(newmodel{h},x_test(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})),y_test);
    Yhat1{h} = predict(newmodel{h},x_test(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})));

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

%% Storage of models for out of cohort validation
save('model_GMV_conf.mat','newmodel','ranks','conf_mdl')
