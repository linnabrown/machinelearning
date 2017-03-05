clear;
clc;
% read data
% trainImages = importdata('trainImages.mat');
% testImages = importdata('testImages.mat');
a = importdata('MnistSet.mat');   
trainLabels = a.train_labels;
testLabels =  a.test_labels; 
K=120;
N=784;
trainLength = length(a.train_images);  
testLength = length(a.test_images); 
testResults = zeros(testLength,K);
trainResults = zeros(trainLength,K);
crossResults = zeros(trainLength,K);
err = zeros(1,K); %numbter of testing error 
err2 = zeros(1,K);%number of 
err3 = zeros(10,K);
trainImages = zeros(196,trainLength);
testImages = zeros(196,testLength);
disp('正在对训练集降维');
for i=1:trainLength
     t =  a.train_images(:,:,i)>127;
     for j=1:14
         for k=1:14
             row = 2*j-1:2*j;
             col = 2*k-1:2*k;
             submatrix = t(row,col);
             trainImages(14*(j-1)+k,i) = sum(submatrix(:))/4;
         end
     end
end
disp('正在对测试集降维');
for i=1:testLength
    t =  a.test_images(:,:,i)>127;
     for j=1:14
         for k=1:14
             row = 2*j-1:2*j;
             col = 2*k-1:2*k;
             submatrix = t(row,col);
             testImages(14*(j-1)+k,i) =sum(submatrix(:))/4;
         end
     end
end
disp('正在对测试集分类');
tic;  
mark=3;
parfor i=1:testLength  
    comp = sum(abs(trainImages - repmat(testImages(:,i),1,trainLength)));  
    [sortedComp,ind] = sort(comp);  
    for j = 1:K  
        ind(j) = trainLabels(ind(j));  %idnex change to the label
        testResults(i,j) = mode(ind(1:j));
    end  
  
end  
% Compute the error rate on the testLength 
disp('计算测试集错误分类率');
for j = 1:K
     err(j) = sum(testResults(:,j) ~= testLabels);
end
%Print out the classification error on the test set  
disp('测试集错误率为：');
err = err/testLength
disp('测试集最佳K值为：');
[~,index] = min(err)
figure;
plot(err);
toc;  
disp(toc-tic);  
disp('正在对训练集分类');
tic;
parfor i=1:trainLength 
    comp = sum(abs(trainImages - repmat(trainImages(:,i),1,trainLength)));  
    [sortedComp,ind] = sort(comp);  
    for j = 1:K  
        ind(j) = trainLabels(ind(j));  %idnex change to the label
        trainResults(i,j) = mode(ind(1:j));
    end  
end  
for j = 1:K
     err2(j) = sum(trainResults(:,j) ~= trainLabels);
end
disp('训练集错误率为：');
err2 = err2/trainLength
disp('训练集最佳K值为：');
err10 = err2;
err10(1) = 1000000;
[~,index2] = min(err10)
figure;
plot(err2);
toc;  
disp(toc-tic); 
%十折交叉验证
disp('十折交叉验证');
tic;
for i = 1:10
    for j = 1:6000
        comp = sum(abs(trainImages - repmat(trainImages(:,6000*(i-1)+j),1,trainLength)));  
        [sortedComp,ind] = sort(comp);
        for k = 1:K  
            ind(k) = trainLabels(ind(k));  
            crossResults(6000*(i-1)+j,k) = mode(ind(1:k));
        end  
    end
    for j = 1:K
        err3(i,j) = sum(crossResults(6000*(i-1)+1:6000*i,j) ~= trainLabels(6000*(i-1)+1:6000*i));
    end
end
% 计算最优K值为index3+1
err10 = err3(:,2:K);
[~,index3] = min(mean(err10));
disp('十折交叉验证最优k值为:');
index3+1
disp('测试集错误率为:');
err(index3+1)
disp('训练集错误率为:');
err2(index3+1)
toc;
disp(toc-tic); 
          