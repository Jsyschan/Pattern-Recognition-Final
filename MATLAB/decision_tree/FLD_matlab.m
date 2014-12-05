function output_data = FLD_matlab( input_data )
%FLD_matlab Performs FLD on the input data

data_dims = size(input_data);
data_classes = input_data(:, data_dims(2));
numClass3 = 0;
numClass4 = 0;
numClass5 = 0;
numClass6 = 0;
numClass7 = 0;
numClass8 = 0;

for i=1:length(input_data)
    if( data_classes(i) == 3 )
        numClass3 = numClass3 + 1;
    elseif( data_classes(i) == 4 )
        numClass4 = numClass4 + 1;
    elseif( data_classes(i) == 5 )
        numClass5 = numClass5 + 1;
    elseif( data_classes(i) == 6 )
        numClass6 = numClass6 + 1;
    elseif( data_classes(i) == 7 )
        numClass7 = numClass7 + 1;
    else
        numClass8 = numClass8 + 1;
    end
end

class3_data = zeros( numClass3, data_dims(2) - 1 );
class4_data = zeros( numClass4, data_dims(2) - 1 );
class5_data = zeros( numClass5, data_dims(2) - 1 );
class6_data = zeros( numClass6, data_dims(2) - 1 );
class7_data = zeros( numClass7, data_dims(2) - 1 );
class8_data = zeros( numClass8, data_dims(2) - 1 );

c3counter = 1;
c4counter = 1;
c5counter = 1;
c6counter = 1;
c7counter = 1;
c8counter = 1;

for i=1:length(input_data)
    if( data_classes(i) == 3 )
        class3_data(c3counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c3counter = c3counter + 1;
    elseif( data_classes(i) == 4 )
        class4_data(c4counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c4counter = c4counter + 1;
    elseif( data_classes(i) == 5 )
        class5_data(c5counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c5counter = c5counter + 1;
    elseif( data_classes(i) == 6 )
        class6_data(c6counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c6counter = c6counter + 1;
    elseif( data_classes(i) == 7 )
        class7_data(c7counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c7counter = c7counter + 1;
    else
        class8_data(c8counter, :) = input_data(i, 1:(data_dims(2) - 1));
        c8counter = c8counter + 1;
    end
end

c3dims = size(class3_data);
c4dims = size(class4_data);
c5dims = size(class5_data);
c6dims = size(class6_data);
c7dims = size(class7_data);
c8dims = size(class8_data);

c3cov = (1 - c3dims(1)) .* cov(class3_data);
c4cov = (1 - c4dims(1)) .* cov(class4_data);
c5cov = (1 - c5dims(1)) .* cov(class5_data);
c6cov = (1 - c6dims(1)) .* cov(class6_data);
c7cov = (1 - c7dims(1)) .* cov(class7_data);
c8cov = (1 - c8dims(1)) .* cov(class8_data);

Sw = c3cov + c4cov + c5cov + c6cov + c7cov + c8cov;

Mt = (1 / length(input_data)) * ( ( numClass3 * mean(class3_data) ) + ( numClass4 * mean(class4_data) ) + ( numClass5 * mean(class5_data) ) + ( numClass6 * mean(class6_data) ) + ( numClass7 * mean(class7_data) ) + ( numClass8 * mean(class8_data) ) );

Sb3 = numClass3 .* ( (mean(class3_data) - Mt) * (mean(class3_data) - Mt)' );
Sb4 = numClass4 .* ( (mean(class4_data) - Mt) * (mean(class4_data) - Mt)' );
Sb5 = numClass5 .* ( (mean(class5_data) - Mt) * (mean(class5_data) - Mt)' );
Sb6 = numClass6 .* ( (mean(class6_data) - Mt) * (mean(class6_data) - Mt)' );
Sb7 = numClass7 .* ( (mean(class7_data) - Mt) * (mean(class7_data) - Mt)' );
Sb8 = numClass8 .* ( (mean(class8_data) - Mt) * (mean(class8_data) - Mt)' );

Sb = Sb3 + Sb4 + Sb5 + Sb6 + Sb7 + Sb8;

output_data = 1;
    

end

