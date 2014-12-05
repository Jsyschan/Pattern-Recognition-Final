clear all;
red = dlmread('../../data/winequality-red.csv', ';', 1, 0);

red_dims = size(red);
red_data = red(:, 1: red_dims(2)-1);
red_classes = red(:,red_dims(2));

load('train_ind.mat');
load('test_ind.mat');

total_train = sum(train_ind);
total_test = sum(test_ind);

red_data_train = zeros(total_train, red_dims(2) - 1);
red_class_train = zeros(total_train, 1);
red_train = zeros(total_train, red_dims(2));

red_data_test = zeros(total_test, red_dims(2) - 1);
red_class_test = zeros(total_test, 1);

counter = 1;
for i=1:red_dims(1)
    if(train_ind(i) == 1)
        red_data_train(counter,:) = red_data(i,:);
        red_class_train(counter) = red_classes(i);
        red_train(counter,:) = red(i,:);
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

results = FLD_matlab(red_train);

% FLDclass = fitcdiscr(red_data_train, red_class_train);

% red_tree = fitctree(reduced_red_data_train, red_class_train);

% red_predictions = predict(FLDclass, red_data_test);

% red_stats = classperf(red_class_test, red_predictions);

% disp('FLD red error rate')
% disp(red_stats.ErrorRate)
