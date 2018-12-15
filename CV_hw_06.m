clc
clear all
close all

run ('./vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup');
tic
%% 1.1
% 1.1.1
trainingDir = './Assignment06_data_reduced/TrainingDataset/';
trainingClass = dir(trainingDir);
trainingClass = trainingClass(3:end);
trainingStruct = struct;

for n=1:length(trainingClass)
    allImg = dir(strcat(trainingDir,trainingClass(n).name,'/*.jpg'));
    fImg = cell(length(allImg),1);
    dImg = cell(length(allImg),1);
   
    for i = 1:length(allImg)
        Img = imread(strcat(trainingDir,trainingClass(n).name,'/',allImg(i).name));
        if(length(size(Img))==3)
            I = single(rgb2gray(Img));
        else
            I = single(Img);
        end
        
        [f, d] = vl_sift(I);
        fImg(i) = {f};
        dImg(i) = {d};
    end
    
    trainingStruct.f.(matlab.lang.makeValidName(trainingClass(n).name)) = fImg;
    trainingStruct.d.(matlab.lang.makeValidName(trainingClass(n).name)) = dImg;
end

% 1.1.2
N = 1000;
cluster = cell(2,1);
Dname = [];
for n=1:length(trainingClass)
    NAME = matlab.lang.makeValidName(trainingClass(n).name);
    Dname = [Dname cell2mat(horzcat(trainingStruct.d.(NAME)'))];
end

[C, A] = vl_kmeans(single(Dname),N, 'algorithm', 'elkan', 'Initialization', 'plusplus');
cluster(:,1) = {C, A};

histgram = cell(length(trainingClass),1);
figure;
for n=1:length(trainingClass)
    Hist = zeros(1,N);
    NAME = matlab.lang.makeValidName(trainingClass(n).name);
    Dname1 = single(cell2mat(horzcat(trainingStruct.d.(NAME)')));
    [index, Dist] = knnsearch(cluster{1,1}',Dname1','distance','euclidean');
    
    [ftshist, binpos] = hist(Dist,15);
    index(Dist > binpos(end-1)) = [];  
    
    for i = 1:length(index)
        Hist(index(i)) = Hist(index(i)) + 1;
    end
    Hist = Hist./length(index);

    subplot(length(trainingClass),1,n);
    plot(Hist);
    title(strcat('Histogram of ',num2str(N),' bins for class: ',trainingClass(n).name));
    histgram(n) = {Hist};
end


testingDir = './Assignment06_data_reduced/TestDataset_';
classCount = zeros(length(histgram));
for n=1:length(histgram)
    imgage = dir(strcat(testingDir,num2str(n)));
    imgage = imgage(3:end);
    for i=1:length(imgage)
        I = rgb2gray(imread(strcat(testingDir,num2str(n),'/',imgage(i).name)));
        [~, d] = vl_sift(single(I));
        [index,Dist] = knnsearch(cell2mat(cluster(1,:))',single(d)','distance','cityblock');
        
        [ftshist,binpos] = hist(Dist,15);
        index(Dist>binpos(end-1)) = [];  
        Hist = zeros(1,1000);
        
        for j = 1:length(index)
            Hist(index(j)) = Hist(index(j)) + 1;
        end
        Hist = Hist./length(index);
        pos = knnsearch(cell2mat(vertcat(histgram)),Hist);
        classCount(n,pos) = classCount(n,pos) + 1;
    end
end
classPercent = classCount./repmat(sum(classCount,2),1,length(histgram))
toc
