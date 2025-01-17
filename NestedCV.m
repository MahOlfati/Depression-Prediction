function [test_idx,train_outer_idx,train_inner_idx,validation_idx] = NestedCV(y,kfold_o,kfold_i,family)
% Nested 10-fold cross-validation considering the family structure
% CV1 : outer fold
% CV2 : inner fold
% Finding unique IDs of family members
[c,ia,ic] = unique(family,'stable');

% Outer folds based on one participant from each family
rng default
CV1 = cvpartition(y(ia),'kfold',kfold_o,'Stratify',true);
for i=1:kfold_o
    test_unique{i} = ia(test(CV1.Impl,i),:); 
    train_outer_unique{i} = ia(training(CV1.Impl,i),:);
end

% Incorporating family members for outer folds
for k=1:kfold_o
  
      [test_idx{k},q] = find(family==family(test_unique{k})');
      [train_outer_idx{k},q] = find(family==family(train_outer_unique{k})');
 
end

% Inner fold based on one participant from each family
for i=1:kfold_o
    rng default
    CV2{i} = cvpartition(y(train_outer_unique{i}),'kfold',kfold_i,'Stratify',true);
    for j = 1:kfold_i
        validation_unique{i,j} = train_outer_unique{i}(test(CV2{i}.Impl,j),:); 
        train_inner_unique{i,j} = train_outer_unique{i}(training(CV2{i}.Impl,j),:);
    end
end

% Incorporating family members for inner folds
for k=1:kfold_o
    for l = 1:kfold_i
      [validation_idx{k,l},q] = find(family==family(validation_unique{k,l})');
      [train_inner_idx{k,l},q] = find(family==family(train_inner_unique{k,l})');
    end
end
end