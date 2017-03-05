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
disp('���ڶ�ѵ������ά');
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
disp('���ڶԲ��Լ���ά');
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
disp('���ڶԲ��Լ�����');
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
disp('������Լ����������');
for j = 1:K
     err(j) = sum(testResults(:,j) ~= testLabels);
end
%Print out the classification error on the test set  
disp('���Լ�������Ϊ��');
err = err/testLength
disp('���Լ����KֵΪ��');
[~,index] = min(err)
figure;
plot(err);
toc;  
disp(toc-tic);  
disp('���ڶ�ѵ��������');
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
disp('ѵ����������Ϊ��');
err2 = err2/trainLength
disp('ѵ�������KֵΪ��');
err10 = err2;
err10(1) = 1000000;
[~,index2] = min(err10)
figure;
plot(err2);
toc;  
disp(toc-tic); 
%ʮ�۽�����֤
disp('ʮ�۽�����֤');
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
% ��������KֵΪindex3+1
err10 = err3(:,2:K);
[~,index3] = min(mean(err10));
disp('ʮ�۽�����֤����kֵΪ:');
index3+1
disp('���Լ�������Ϊ:');
err(index3+1)
disp('ѵ����������Ϊ:');
err2(index3+1)
toc;
disp(toc-tic); 
          