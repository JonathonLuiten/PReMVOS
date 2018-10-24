function [] = run(folder)

curr_files = dir ([folder '/*.jpg']);
num_files = numel (curr_files);
rng('shuffle')

if num_files == 2500
    return
end

addpath(genpath('patch-inpainting'));
addpath(genpath('patch-inpainting/CSH_code'));
addpath(genpath('PoissonEdiitng'));
addpath(genpath('PoissonEdiitng/PoissonEdiitng20151105'));
addpath(genpath('PoissonEdiitng/PoissonEdiitng20151105/src'));
addpath(genpath('src'));

img =imread([folder '/0.jpg']);
[gt,map] = imread([folder '/0.png']);
outpath = folder;

for i = 1:2499
    t = tic;
    [im1, gt1, ~]= lucid_dream(img,gt,0);
    imwrite(im1,sprintf('%s%d.jpg',outpath,i))
    imwrite(gt1,map,sprintf('%s%d.png',outpath,i))
    disp(i)
    disp(toc(t))
end
