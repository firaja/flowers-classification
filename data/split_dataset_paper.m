% MATLAB script used to obtain the same dataset subdivision of the original paper
% Data can be found here http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

ds = imageDatastore('jpg');

for i = 1:102
    mkdir(['train/' int2str(i)]);
    mkdir(['valid/' int2str(i)]);
    mkdir(['test/' int2str(i)]);
end

load('imagelabels.mat');
load('setid.mat');

ds.Labels = labels;

for i = 1:size(trnid, 2)
    img_name = split(ds.Files{trnid(i)}, '/');
    img_name = img_name{end};
    movefile(['jpg/' img_name], ['train/' int2str(ds.Labels(trnid(i))) '/' img_name])
end

for i = 1:size(valid, 2)
    img_name = split(ds.Files{valid(i)}, '/');
    img_name = img_name{end};
    movefile(['jpg/' img_name], ['valid/' int2str(ds.Labels(valid(i))) '/' img_name])
end

for i = 1:size(tstid, 2)
    img_name = split(ds.Files{tstid(i)}, '/');
    img_name = img_name{end};
    movefile(['jpg/' img_name], ['test/' int2str(ds.Labels(tstid(i))) '/' img_name])
end