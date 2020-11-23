clear;close all;clc;

N=1000; n = 2; K=10;
[traindata,trainlabels] = generateMultiringDataset(n,N);

x = traindata;
l = 2*(trainlabels - 1 - 0.5);

N2=10000; n = 2; K=10;
[testdata,testlabels] = generateMultiringDataset(n,N2);

x2 = testdata;
l2 = 2*(testlabels - 1 - 0.5);
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
figure(2), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);

d = SVMBest.predict(x2')'; % Labels of testing data using the trained SVM
indINCORRECT = find(l2.*d == -1); % Find testing samples that are incorrectly classified by the trained SVM
indCORRECT = find(l2.*d == 1); % Find testing samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x2(1,indCORRECT),x2(2,indCORRECT),'g.'), hold on,
plot(x2(1,indINCORRECT),x2(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTestingError = length(indINCORRECT)/N2, % Empirical estimate of training error probability
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,[0 0]); xlabel('x1'), ylabel('x2'), axis equal,
