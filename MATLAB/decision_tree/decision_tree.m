red = dlmread('../../data/winequality-red.csv', ';', 1, 0);
white = dlmread('../../data/winequality-white.csv', ';', 1, 0);

red_dims = size(red);
red_data = red(:, 1: red_dims(2)-1);
red_classes = red(:,red_dims(2));

load('train_ind.mat');
load('test_ind.mat');

total_train = sum(train_ind);
total_test = sum(test_ind);

red_data_train = zeros(total_train, red_dims(2) - 1);
red_class_train = zeros(total_train, 1);

red_data_test = zeros(total_test, red_dims(2) - 1);
red_class_test = zeros(total_test, 1);

counter = 1;
for i=1:red_dims(1)
    if(train_ind(i) == 1)
        red_data_train(counter,:) = red_data(i,:);
        red_class_train(counter) = red_classes(i);
        counter = counter + 1;
    end
end

counter = 1;
for i=1:red_dims(1)
    if(test_ind(i) == 1)
        red_data_test(counter,:) = red_data(i,:);
        red_class_test(counter) = red_classes(i);
        counter = counter + 1;
    end
end

% red_train_cutoff = floor(0.9 * length(red_data));
% red_data_train = red_data(1:red_train_cutoff, :);
% red_classes_train = red_classes(1:red_train_cutoff, :);

% red_data_test = red_data(red_train_cutoff + 1:length(red), :);
% red_classes_test = red_classes(red_train_cutoff + 1:length(red), :);

white_dims = size(white);
white_data = white(:, 1: white_dims(2)-1);
white_classes = white(:,white_dims(2));

%[white_train_ind, white_test_ind] = crossvalind('HoldOut', length(white), 0.3);

load('white_train_ind.mat');
load('white_test_ind.mat');

total_train_white = sum(white_train_ind);
total_test_white = sum(white_test_ind);

% white_train_cutoff = floor(0.9 * length(white_data));
% white_data_train = white_data(1:white_train_cutoff, :);
% white_class_train = white_classes(1:white_train_cutoff, :);

% white_data_test = white_data(white_train_cutoff + 1:length(white), :);
% white_class_test = white_classes(white_train_cutoff + 1:length(white), :);

white_data_train = zeros(total_train_white, white_dims(2) - 1);
white_class_train = zeros(total_train_white, 1);

white_data_test = zeros(total_test_white, white_dims(2) - 1);
white_class_test = zeros(total_test_white, 1);

counter = 1;
for i=1:white_dims(1)
    if(white_train_ind(i) == 1)
        white_data_train(counter,:) = white_data(i,:);
        white_class_train(counter) = white_classes(i);
        counter = counter + 1;
    end
end

counter = 1;
for i=1:white_dims(1)
    if(white_test_ind(i) == 1)
        white_data_test(counter,:) = white_data(i,:);
        white_class_test(counter) = white_classes(i);
        counter = counter + 1;
    end
end

red_tree = fitctree(red_data_train, red_class_train);
white_tree = fitctree(white_data_train, white_class_train);

red_predictions = predict(red_tree, red_data_test);
white_predictions = predict(white_tree, white_data_test);

red_stats = classperf(red_class_test, red_predictions);
white_stats = classperf(white_class_test, white_predictions);

disp('Red error rate')
disp(red_stats.ErrorRate)
disp('White error rate')
disp(white_stats.ErrorRate)