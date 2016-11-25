% Eigenfaces Facial Recognition 
% CSCI 4830 Computer Vision Final Project
%
% Thomas Lillis
% James Waugh


% Open ORL database of faces
subjects = dir('orl_faces/*');

NUMBER_OF_SUBJECTS = 40;
IMAGES_PER_SUBJECT = 10;
dataset = cell(NUMBER_OF_SUBJECTS, IMAGES_PER_SUBJECT);

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
            end
        end
    end
end

% Example indexing dataset
subject_number = 5;
image_number = 8;

imshow(dataset{subject_number,image_number})

