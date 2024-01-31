% ----------------------------------------------------------------------- %
%            R E P L I C A T I O N     P R E D I C T I O N                %
% ----------------------------------------------------------------------- %
% This script predicts depressive symptoms of Achenbach questionnaire     % 
% in out of cohort dataset using pre-trained models on HCP dataset.       %
%                                                                         %
%   Input parameters:                                                     %
%       - pre_trained:      Pre-trained model contains models and their   %
%                           features' rank.                               %
%       - vol:              473 GMV or fALFF or ReHo features.            %
%       - depression:       Achenbach questionnaire scores.               %
%       - sleep:            Sleep quality parameters.                     %
%       - confounding:      Age,gender, ant tGMV considered as confound   %
%                           variables.                                    %
%                                                                         %
%   Output variables:                                                     %
%       - YHat_cohort:         Predicted values.                          %
%       - R2:               Coefficient of determination.                 %
%       - perf:             Mean absolute error.                          %
%       - ci:               95% confidence interval.                      %
%       - MSE:              Mean squared error.                           %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:          1.2.                                          %
%       - Author:           Mahnaz Olfati                                 %
%       - Date:             21/02/2022                                    %
% ----------------------------------------------------------------------- %
% Read data
clear,close,clc

pre_trained = load('model_*.mat');        % Put model, which has been saved by sleep_predictor.m or GMV_predictor.m file, here

% Initializing parameters
sleep                  % Put sleep variable of the cohort here
depression             % Put depression varible of the cohort here 
confounding            % Put confounding variablesof the cohort here
anxi                   % Put anxiety variable of the cohort here
vol                    % Put GMV variable of the cohort here

%% Predictor and target vriables
x = sleep;             % Put predictors here
y = depression;        % Put target here

%% Mean prediction of new data based on pre-trained models
feature_number = zeros(1,size(pre_trained.newmodel,2));
yhat_cohort = zeros(size(target,1),size(pre_trained.newmodel,2));
loss_cohort = zeros(1,size(pre_trained.newmodel,2));
 
% prediction of target based on all pre-trained models
for model = 1:size(pre_trained.newmodel,2)
    
    % confound removal based on pre-trained models
    predictor = zeros(size(x));
    for col = 1:size(x,2)
        predictor(:,col) = x(:,col) - predict(pre_trained.conf_mdl{model,col},confounding);
    end
    
    % prediction of targets based on pre-trained models
    feature_number(1,model) = size(pre_trained.newmodel{model}.X,2);
    yhat_cohort(:,model) = predict(pre_trained.newmodel{model},predictor(:,pre_trained.ranks{model}(1:feature_number(1,model))));
    loss_cohort(1,model) = loss(pre_trained.newmodel{model},predictor(:,pre_trained.ranks{model}(1:feature_number(1,model))),target);

end

% mean prediction
YHat_cohort = mean(yhat_cohort,2);

%% Plotting
reg = [YHat_cohort,target];
figure;
[R,PValue] = corrplot(reg,'testR','on')

R2 = 1 - sum((target - YHat_cohort) .^ 2) / sum((target - mean(target)) .^ 2);

e = YHat_cohort-target;
perf = mae(e);

% 95% confidence interval
ts = tinv([0.025 0.975],length(YHat_cohort)-1);
sem = std(YHat_cohort)/sqrt(length(YHat_cohort));
ci = mean(YHat_cohort)+ts*sem

MSE = mean(loss_cohort)

%% Storage of results
save('sleep_hcpaging.mat')
