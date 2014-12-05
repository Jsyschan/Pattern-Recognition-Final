white = dlmread('../../data/winequality-white.csv', ';', 1, 0);

[white_train_ind, white_test_ind] = crossvalind('HoldOut', length(white), 0.3);