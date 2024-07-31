function [x_train,x_test,conf_mdl] = Confound_Remove_model(outer,x_train,x_test,conftrain_outer,conftest,conf_mdl)

    for culm = 1:size(x_train,2)
            conf_mdl{outer,culm} = fitlm(conftrain_outer, x_train(:,culm));
            x_train(:,culm) = table2array(conf_mdl{outer,culm}.Residuals(:,1));
            x_test(:,culm) = x_test(:,culm) - predict(conf_mdl{outer,culm},conftest);
    end
    
end