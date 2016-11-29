% Eigenfaces Facial Recognition 
% CSCI 4830 Computer Vision Final Project
%
% Thomas Lillis
% James Waugh

clear all;
close all;


% Open ORL database of faces
subjects = dir('orl_faces/*');

% static values
NUMBER_OF_SUBJECTS = 40;
IMAGES_PER_SUBJECT = 10;
NUMBER_OF_IMAGES = NUMBER_OF_SUBJECTS * IMAGES_PER_SUBJECT;
IMAGE_SCALE = .3;

% containers for data
dataset = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT);
dataset_v = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT); % can probably get rid of this?
vec = cell(NUMBER_OF_IMAGES);

image_num = 1;

for subject = subjects'
    subject_number = sscanf(subject.name,'s%d');
    if(subject_number)
        sub_dir = strcat('orl_faces/',subject.name,'/*');
        images = dir(sub_dir);
        for image = images'
            if(~strcmp(image.name,'.') && ~strcmp(image.name,'..'))
                image_path = strcat('orl_faces/',subject.name,'/',image.name);
                image_number = sscanf(image.name,'%d.pgm');
                dataset(subject_number,image_number)={imresize(imread(image_path),IMAGE_SCALE)};
                dataset_v(subject_number,image_number)={dataset{subject_number,image_number}(:).'}; % Not needed?
                vec(image_num) = dataset_v(subject_number,image_number);
                image_num = image_num + 1;
            end
        end
    end
end

% Calculate eigenfaces

[h,w] = size(dataset{1,1}); % Get size of image

% Put into matrix of a known size
x = zeros(h*w,NUMBER_OF_IMAGES);
for image = 1:NUMBER_OF_IMAGES
   x(:,image) = vec{image}; 
end
x = double(x);

% Subtract mean
x = bsxfun(@minus, x, mean(x,2));

% calculate covariance
s = cov(x');

% obtain eigenvalue & eigenvector
[V,D] = eig(s);
eigval = diag(D);

% sort eigenvalues in descending order
eigval = eigval(end:-1:1);
V = fliplr(V);

% show 0th through 15th principal eigenvectors
eig0 = repmat(mean(x,2), [h,w]);
figure,subplot(4,4,1)
imagesc(eig0)
colormap gray
for k = 1:15
    subplot(4,4,k+1)
    imagesc(reshape(V(:,k),h,w))
end

