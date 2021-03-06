% Eigenfaces Facial Recognition 
% CSCI 4830 Computer Vision Final Project
%
% Thomas Lillis
% James Waugh

%clear all;
close all;
%% Load Data Set

% Open ORL database of faces
subjects = dir('orl_faces/*');

% static values
NUMBER_OF_SUBJECTS = 40;
IMAGES_PER_SUBJECT = 10;
NUMBER_OF_IMAGES = NUMBER_OF_SUBJECTS * IMAGES_PER_SUBJECT;
IMAGE_SCALE = .5;

% containers for data
dataset = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT);
dataset_v = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT);
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
                dataset_v(subject_number,image_number)={dataset{subject_number,image_number}(:).'};
                vec(image_num) = {dataset{subject_number,image_number}(:).'};
                image_num = image_num + 1;
            end
        end
    end
end

correct = 0;
incorrect = 0;

%% Cross Validition
for test_image = 1:NUMBER_OF_IMAGES-1

    %% Calculate eigenfaces

    [h,w] = size(dataset{1,1}); % Get size of image
    d= h*w;
    % Put into matrix of a known size
    x = zeros(d,NUMBER_OF_IMAGES-1);
    offset = 0;
    for image = 1:NUMBER_OF_IMAGES
        if image == test_image
            offset = 1;
            image = image + 1;
        end
       x(:,image-offset) = vec{image}; 
    end
    x = double(x);

    % Subtract mean
    m = mean(x,2);
    x = bsxfun(@minus, x, m);

    % calculate covariance
    s = cov(x');

    % obtain eigenvalue & eigenvector
    [V,D] = eig(s);
    eigval = diag(D);

    % sort eigenvalues in descending order
    eigval = eigval(end:-1:1);
    V = fliplr(V);

    % show 0th through 15th principal eigenvectors
%     eig0 = repmat(m, [h,w]);
%     figure,subplot(4,4,1)
%     %imagesc(eig0)
%     colormap gray
%     for k = 1:16
%         subplot(4,4,k)
%         imagesc(reshape(V(:,k),h,w))
%     end

    eigsum= sum(eigval);
    csum= 0;
    for i= 1:d
        csum= csum + eigval(i);
        tv= csum/eigsum;
        if tv>0.95
            k95= i;
            break;
        end
    end

    %% Identify new face

    % Load new face image and manipulate into column vector
    % For now just taking image from loaded dataset
    face = double(vec{test_image})';

    % Compute the weights for every image
    wn = double(zeros(k95,1));
    Wm = zeros(NUMBER_OF_IMAGES-1,k95);
    for i = 1:k95 
       wn(i) = V(:,i)'*(face-m);
       offset = 0;
       for j = 1:NUMBER_OF_IMAGES
           if j == test_image
                offset = 1;
                j = j + 1;
            end
           Wm(j-offset,i) = V(:,i)'*((double(vec{j})')-m);
       end
    end

    d = zeros(NUMBER_OF_IMAGES-1,1);
    for j = 1:NUMBER_OF_IMAGES-1
        d(j) = norm(wn-Wm(j,:)');
    end

    [M,I] = min(d);
    
    if(I > test_image)
       I = I + 1; 
    end
    
    if(floor((I-1)/10) == floor((test_image-1)/10))
        correct = correct + 1;
%         figure;
%         subplot(1,2,1)
%         imshow(uint8(reshape(vec{test_image},h,w)))
%         title('Test face')
%         subplot(1,2,2)
%         imshow(uint8(reshape(vec{I},h,w)))
%         title('Recognized face')
    else
        fprintf('Face incorrectly classified!\n')
        fprintf('--Got image number:        %i\n', I)
        fprintf('--Test image number:       %i\n', test_image)
        fprintf('--Got image subject:       %i\n', floor((I-1)/IMAGES_PER_SUBJECT))
        fprintf('--Correct image subject:   %i\n', floor((test_image-1)/IMAGES_PER_SUBJECT))
        incorrect = incorrect + 1;
        figure;
        subplot(1,2,1)
        imshow(uint8(reshape(vec{test_image},h,w)))
        title('Test face')
        subplot(1,2,2)
        imshow(uint8(reshape(vec{I},h,w)))
        title('Recognized face')
    end
end

fprintf('Number correctly classified:   %i\n', correct)
fprintf('Number incorrectly classified: %i\n', incorrect)
fprintf('Percent correctly classified:  %f\n', correct/(correct+incorrect)*100)