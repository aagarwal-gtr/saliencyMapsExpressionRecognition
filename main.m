img_url_s = 'C:/Users/ABHINAV/Documents/MATLAB/dataset/3_s.png';
img_url_h = 'C:/Users/ABHINAV/Documents/MATLAB/dataset/3_h.png';
%img_url3 = 'C:/Users/ABHINAV/Documents/MATLAB/neutral_haar/2_.png';

p = default_fast_param;
p.blurRadius = 0.02;

img_s = imread(img_url_s);
map_s = simpsal(img_s,p);

img_h = imread(img_url_h);
map_h = simpsal(img_h,p);

%img3 = imread(img_url3);
%map_n = simpsal(img3,p);

subplot(2,3,1);
imshow(img_s);
title('Image surprise');

subplot(2,3,2);
imshow(map_s);
title('Saliency map surprise');

subplot(2,3,4);
imshow(img_h);
title('Image happy');

subplot(2,3,5);
imshow(map_h);
title('Saliency map happy');

%subplot(1,3,4);
%imshow(map_n);
%title('Saliency map neutral');

%subplot(2,3,4);
%subplot(2,3,6);

path = 'C:\Users\ABHINAV\Documents\MATLAB\dataset';
dataset = ls('C:\Users\ABHINAV\Documents\MATLAB\dataset');
%dataset = ls(path);
dataset = cellstr(dataset);
dataset = dataset(3:174);
count_dataset = 172;
features=[];
labels=[];
for(i=1:172)
    path = 'C:\Users\ABHINAV\Documents\MATLAB\dataset';
    path = strcat(path, '\');
    path = strcat(path, dataset{i});
    p = default_fast_param;
    p.blurRadius = 0.02;
    img = imread(path);
    map = simpsal(img,p);
    features = [features, map(:)];
    %h = 1, s = 0
    if(isempty(findstr(dataset{i}, 'h.png')))
        value = 0;
    else
        value = 1;
    end
    labels = [labels, value];
end

num_points = size(features, 2);
split_point = round(num_points*0.7);
seq = randperm(num_points);
X_train = features(:,seq(1:split_point)); %features of training set
Y_train = labels(seq(1:split_point)); %labels of training set
X_test = features(:,seq(split_point+1:end)); %features of testing set
Y_test = labels(seq(split_point+1:end)); %labels of testing set

SVMModel = svmtrain(transpose(X_train), transpose(Y_train));
result = svmclassify(SVMModel, transpose(X_test));
count_right = 0;
Y_testt = transpose(Y_test);
for(i=1:numel(Y_testt))
    if(result(i) == Y_testt(i))
        count_right = count_right + 1;
    end
end