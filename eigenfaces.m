% Eigenfaces Facial Recognition 
% CSCI 4830 Computer Vision Final Project
%
% Thomas Lillis
% James Waugh

clear all;
close all;


% Open ORL database of faces
subjects = dir('orl_faces/*');

NUMBER_OF_SUBJECTS = 40;
IMAGES_PER_SUBJECT = 10;
dataset = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT);
dataset_v = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT); % vector form for performing eigenfaces

for subject = subjects'
    subject_number = sscanf(subject.name,'s%d');
    if(subject_number)
        sub_dir = strcat('orl_faces/',subject.name,'/*');
        images = dir(sub_dir);
        for image = images'
            if(~strcmp(image.name,'.') && ~strcmp(image.name,'..'))
                image_path = strcat('orl_faces/',subject.name,'/',image.name);
                image_number = sscanf(image.name,'%d.pgm');
                dataset(subject_number,image_number)={imread(image_path)};
                dataset_v(subject_number,image_number)={dataset{subject_number,image_number}(:).'};
            end
        end
    end
end

% Calculate eigenfaces

for subject=1:1%NUMBER_OF_SUBJECTS
   for image=1:1%IMAGES_PER_SUBJECT
       x = dataset_v{subject,image};
       [h,w,n] = size(dataset{subject,image});
       x = double(x);
       x = bsxfun(@minus, x, mean(x,2));
       s = cov(x');
       [V,D] = eig(s);
       eigval = diag(D);
       % sort eigenvalues in descending order
       eigval = eigval(end:-1:1);
       V = fliplr(V);
       % show 0th through 15th principal eigenvectors
       eig0 = reshape(mean(x,2), [h,w]);
       % eig0 = repmat(mean(x,2), [h,w]);
       figure,subplot(4,4,1)
       imagesc(eig0)
       colormap gray
       for k = 1:15
           subplot(4,4,k+1)
           imagesc(reshape(V(:,k),h,w))
       end
   end
end

% Example indexing dataset
subject_number = 5;
image_number = 8;

imshow(dataset{subject_number,image_number})

