function [Data Labels Files Objs Matches] = loadDomains(PARAM)

base_dir = PARAM.base_dir;
domains = PARAM.domains;
histfile = PARAM.histfile;
image_dirs = PARAM.image_dirs;
categories = PARAM.categories;
numclasses = length(PARAM.categories);

% Load data
Objs = cell(size(domains));
for i = 1:length(domains)
    disp(sprintf('Loading %s...', domains{i}));
    curr = 1;
    for j = 1:numclasses
        files = dir([base_dir domains{i} categories{j} '/' histfile{i}]);
        for k = 1:length(files)
            clear histogram object_id
            load([base_dir domains{i} categories{j} '/' files(k).name], 'histogram');
            v = histogram;
            %normalize histogram
            v = v / norm(v);
            
            if ~isempty(strfind(domains{i}, 'webcam')) | ~isempty(strfind(domains{i}, 'dslr'))
                load([base_dir domains{i} categories{j} '/' files(k).name], 'object_id');
                Objs{i}(curr) = str2num(object_id);
            end
            
            Data{i}(curr,:) = v;
            Labels{i}(curr) = j;
            Files{i}{curr} = [image_dirs{i} categories{j} ...
                '/' 'frame' files(k).name(10:14) '.jpg'];
            curr = curr + 1;
        end
    end
end

Matches = loadCorrespondences(Files);

