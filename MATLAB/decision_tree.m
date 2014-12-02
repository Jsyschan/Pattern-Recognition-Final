red = dlmread('../data/winequality-red.csv', ';', 1, 0);
white = dlmread('../data/winequality-white.csv', ';', 1, 0);

red_dims = size(red);
red_data = red(:, 1: red_dims(2)-1);
red_classes = red(:,red_dims(2));

red_train_cutoff = floor(0.9 * length(red_data));
red_data_train = red_data(1:red_train_cutoff, :);
red_classes_train = red_classes(1:red_train_cutoff, :);

red_data_test = red_data(red_train_cutoff + 1:length(red), :);
red_classes_test = red_classes(red_train_cutoff + 1:length(red), :);

white_dims = size(white);
white_data = white(:, 1: white_dims(2)-1);
white_classes = white(:,white_dims(2));

white_train_cutoff = floor(0.9 * length(white_data));
white_data_train = white_data(1:white_train_cutoff, :);
white_classes_train = white_classes(1:white_train_cutoff, :);

white_data_test = white_data(white_train_cutoff + 1:length(white), :);
white_classes_test = white_classes(white_train_cutoff + 1:length(white), :);

red_tree = fitctree(red_data_train, red_classes_train);
white_tree = fitctree(white_data_train, white_classes_train);

red_predictions = predict(red_tree, red_data_test);
white_predictions = predict(white_tree, white_data_test);

red_stats = classperf(red_classes_test, red_predictions);
white_stats = classperf(white_classes_test, white_predictions);

disp('Red error rate')
display(red_stats.ErrorRate)
disp('White error rate')
display(white_stats.ErrorRate)