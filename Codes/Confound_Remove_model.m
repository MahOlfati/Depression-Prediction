function [x_train,x_test,conf_mdl] = Confound_Remove_model(h,x_train,x_test,conftrain_outer,conftest)

for culm = 1:size(x_train,2)
        conf_mdl{h,culm} = fitlm(conftrain_outer, x_train(:,culm));
        x_train(:,culm) = table2array(conf_mdl{h,culm}.Residuals(:,1));
        x_test(:,culm) = x_test(:,culm) - predict(conf_mdl{h,culm},conftest);
end
end