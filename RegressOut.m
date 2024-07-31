function [x_trainn,x_validation] = RegressOut(outer,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num)

    for culmn = 1:feature_num
            x_trainn(:,culmn) = x_trainn(:,culmn) - predict(conf_mdl{outer,ranks{outer}(culmn)},conftrain_inner);
            x_validation(:,culmn) = x_validation(:,culmn) - predict(conf_mdl{outer,ranks{outer}(culmn)},conf_validation);
    end
    
end