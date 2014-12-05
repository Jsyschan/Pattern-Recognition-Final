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

red_coeff = pca(red_data_train);

% dimReduc5 = red_coeff(:, 1:5);
dimReduc3 = red_coeff(:, 1:3);

reduced_red_data_train = red_data_train * dimReduc3;
reduced_red_data_test = red_data_test * dimReduc3;

net = feedforwardnet(10, 'traingd');
[net, tr] = train(net, reduced_red_data_train', red_class_train');

resultRaw = net(reduced_red_data_test');
result = resultRaw;

for i=1:length(resultRaw)
    result(i) = round(resultRaw(i));
end

numErr = 0;
for i=1:length(result)
    if(result(i) ~= red_class_test(i))
        numErr = numErr + 1;
    end
end

numErr / length(result)