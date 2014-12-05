tic;
red = dlmread('../../data/winequality-red.csv', ';', 1, 0);

red_dims = size(red);
red_data = red(:, 1: red_dims(2)-1);
red_classes = red(:,red_dims(2));

red_dev = std(red_data);
red_mu = mean(red_data);

for i=1:red_dims(2)-1
    red_data(:,i) = normpdf(red_data(:,i), red_mu(i), red_dev(i));
end

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

red_class_3_train = red_class_train;
for i=1:length(red_class_3_train)
    if( red_class_3_train(i) == 3 )
        red_class_3_train(i) = 1;
    else
        red_class_3_train(i) = 0;
    end
end

red_class_4_train = red_class_train;
for i=1:length(red_class_4_train)
    if( red_class_4_train(i) == 4 )
        red_class_4_train(i) = 1;
    else
        red_class_4_train(i) = 0;
    end
end

red_class_5_train = red_class_train;
for i=1:length(red_class_5_train)
    if( red_class_5_train(i) == 5 )
        red_class_5_train(i) = 1;
    else
        red_class_5_train(i) = 0;
    end
end

red_class_6_train = red_class_train;
for i=1:length(red_class_6_train)
    if( red_class_6_train(i) == 6 )
        red_class_6_train(i) = 1;
    else
        red_class_6_train(i) = 0;
    end
end

red_class_7_train = red_class_train;
for i=1:length(red_class_7_train)
    if( red_class_7_train(i) == 7 )
        red_class_7_train(i) = 1;
    else
        red_class_7_train(i) = 0;
    end
end

red_class_8_train = red_class_train;
for i=1:length(red_class_8_train)
    if( red_class_8_train(i) == 8 )
        red_class_8_train(i) = 1;
    else
        red_class_8_train(i) = 0;
    end
end

disp('Creating SVM classifiers');
red3Classifier = fitcsvm(red_data_train, red_class_3_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('3 Done');
red4Classifier = fitcsvm(red_data_train, red_class_4_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('4 Done');
red5Classifier = fitcsvm(red_data_train, red_class_5_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('5 Done');
red6Classifier = fitcsvm(red_data_train, red_class_6_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('6 Done');
red7Classifier = fitcsvm(red_data_train, red_class_7_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('7 Done');
red8Classifier = fitcsvm(red_data_train, red_class_8_train, 'KernelScale', 'auto', 'KernelFunction', 'polynomial', 'Standardize', true);
disp('8 Done');

disp('Predicting labels');
[label3, score3] = predict(red3Classifier, red_data_test);
disp('3 Done');
[label4, score4] = predict(red4Classifier, red_data_test);
disp('4 Done');
[label5, score5] = predict(red5Classifier, red_data_test);
disp('5 Done');
[label6, score6] = predict(red6Classifier, red_data_test);
disp('6 Done');
[label7, score7] = predict(red7Classifier, red_data_test);
disp('7 Done');
[label8, score8] = predict(red8Classifier, red_data_test);
disp('8 Done');

numErr = 0;
for i=1:length(red_class_test)
    if( red_class_test(i) == 3 )
        if( label3(i) == 0 )
            numErr = numErr + 1;
        end
    elseif( red_class_test(i) == 4 )
        if( label4(i) == 0 )
            numErr = numErr + 1;
        end
    elseif( red_class_test(i) == 5 )
        if( label5(i) == 0 )
            numErr = numErr + 1;
        end
    elseif( red_class_test(i) == 6 )
        if( label6(i) == 0 )
            numErr = numErr + 1;
        end
    elseif( red_class_test(i) == 7 )
        if( label7(i) == 0 )
            numErr = numErr + 1;
        end
    elseif( red_class_test(i) == 8 )
        if( label8(i) == 0 )
            numErr = numErr + 1;
        end
    end
end

numErr / length(red_class_test)
toc