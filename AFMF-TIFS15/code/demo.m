clear;
addpath('kerest'); % for kernel estimation
addpath('measures'); % scalar-based median filtering forensics

gray_img_dir = 'grayscale_dataset/';
mf_img_dir = 'mf_dataset/';

    
exist("grayscale_dataset/" , 'dir');
exist("mf_dataset", 'dir');

file_list= dir(fullfile("mf_dataset/", '*.png'));
% gray_image = dir(fullfile("grayscale_dataset/", '*png'))
nfiles = length(file_list);
    opts.winS = 3;
    opts.kernel = [];
    opts.init_kernel = fspecial('average',opts.winS); % AVEE kernel
    opts.iDither = true; % pixel value perturbation
    opts.nb_omega = 0.1;
    opts.nb_lambda = 1500;
    opts.nb_gamma = 500;
    
exist("grayscale_dataset/" , 'dir');
exist("mf_dataset", 'dir');

file_list= dir(fullfile("mf_dataset/", '*.png'));
nfiles = length(file_list);

for ii = 1:nfiles
% for ii = 1:4
    curr_file_name = file_list(ii).name;
    fprintf("%s \n", curr_file_name);

    grayscale_file = fullfile("grayscale_dataset/", curr_file_name);
    mf_file = fullfile("mf_dataset/", curr_file_name);
   
    oriITC = imread(grayscale_file);
    mfITC = imread(mf_file);
    oriI = oriITC(:,:,1); % Extract the first channel
    mfI = mfITC(:,:,1);
    % imshow(oriI);
    % imshow(oriITC);
    afI = uint8(MFDCONV(mfI, opts)); % MF anti-forensic image
    mfdc_dir = "mfdc_dataset/";
    mfdc__img_path = strcat(mfdc_dir, curr_file_name);
    axis off;
    imwrite(afI, mfdc__img_path);
    

    % results
    % fprintf('oriI\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n',psnr(oriI,oriI),ssim(oriI,oriI),kirchnerSPIE10(oriI,1,0),kirchnerSPIE10(oriI,1,0,64),caoICME10(oriI,7,100),mff44(oriI,true));
    % fprintf('mfI\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n',psnr(oriI,mfI),ssim(oriI,mfI),kirchnerSPIE10(mfI,1,0),kirchnerSPIE10(mfI,1,0,64),caoICME10(mfI,7,100),mff44(mfI,true));
    % fprintf('afI\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n',psnr(oriI,afI),ssim(oriI,afI),kirchnerSPIE10(afI,1,0),kirchnerSPIE10(afI,1,0,64),caoICME10(afI,7,100),mff44(afI,true));


end   

