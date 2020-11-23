clear;close all;clc;

%%Generate Data
numberOfTraining = 1000;
train = exam4q1_generateData(numberOfTraining);
x1_train = train(1,:);
x2_train = train(2,:);

numberOfTesting = 10000;
test = exam4q1_generateData(numberOfTesting);
x1_test = test(1,:);
x2_test = test(2,:);

%Choosing the maximum number of perceptrons
perceptrons = 10;
%Choosing the number of folds
fold = 10;

PS = struct();%Store the parameters

%Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,numberOfTraining,fold+1));
for k = 1:fold
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Determine/specify sizes of parameter matrices/vectors
nX =size(x1_train,1);%1; 
nY =size(x2_train,1);%;1

%% LOOP
for nPerceptrons = 1:perceptrons
    
    sizeParams = [nX;nPerceptrons;nY]
    j=1;
    
    %K-Fold Cross-Validation
    for i = 1:fold
        
        indValidate = [indPartitionLimits(i,1):indPartitionLimits(i,2)];
        %Using fold k as validation set
        xValidate = x1_train(:,indValidate);
        yValidate = x2_train(:,indValidate);
        if i == 1
            indTrain = [indPartitionLimits(i,2)+1:numberOfTraining];
        elseif i == fold
            indTrain = [1:indPartitionLimits(i,1)-1];
        else
            indTrain = [1:indPartitionLimits(i-1,2),indPartitionLimits(i+1,1):numberOfTraining];
        end
        
        %Using all other folds as training set
        xTrain = x1_train(:,indTrain);
        yTrain = x2_train(:,indTrain);
        Ntrain = length(indTrain); Nvalidate = length(indValidate);
        
        % Initialize model parameters
        params.A = randn(nPerceptrons,nX);
        params.b = rand(nPerceptrons,1);
        params.C = rand(nY,nPerceptrons);
        params.d = mean(x2_train,2);%zeros(nY,1); % initialize to mean of y
        %params = paramsTrue;
        vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
        %vecParamsInit = vecParamsTrue; % Override init weights with true weights
        
        % Optimize model
        options = optimset('MaxFunEvals',100000,'MaxIter',100000);
        vecParams = fminsearch(@(vecParams)(objectiveFunction(xTrain,yTrain,sizeParams,vecParams,j)),vecParamsInit,options);
        
        % Visualize model output for training data
        params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
        params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
        params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
        params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
        
        
        H = mlpModel(xValidate,params,j);
        
        %         Er(nPerceptrons,i)=Error(yValidate,H);        %Calculate overall error per fold
        %         if j==1
        %             avgError1(nPerceptrons)=mean(Er(nPerceptrons,:));          %average fold error
        %         else
        %             avgError2(nPerceptrons)=mean(Er(nPerceptrons,:));          %average fold error
        %         end
        MSE(i)=sum((yValidate-H).^2)/size(yValidate,2); %error given parameters
    end
    PS(j,nPerceptrons).A = params.A;
    PS(j,nPerceptrons).b = params.b;
    PS(j,nPerceptrons).C = params.C;
    PS(j,nPerceptrons).d = params.d;
    
    Error(nPerceptrons)=mean(MSE); %average error for each fold
    nPerceptrons = nPerceptrons + 1;
end

%% Find best combination of model order and actiation nonlinearity
[val,idx] = min(Error(:));
[row,col] = ind2sub(size(Error),idx);

%% Train an MLP with specification using Dtrain
% Determine/specify sizes of parameter matrices/vectors
nX =size(x1_train,1);%1; 
nY =size(x2_train,1);%;1
nPerceptrons = col;
sizeParams = [nX;nPerceptrons;nY];

% Initialize model parameters
params.A = PS(row,nPerceptrons).A;%zeros(nPerceptrons,nX);
params.b = PS(row,nPerceptrons).b;%zeros(nPerceptrons,1);
params.C = PS(row,nPerceptrons).C;%zeros(nY,nPerceptrons);
params.d = PS(row,nPerceptrons).d;%mean(Y,2);%zeros(nY,1); % initialize to mean of y

vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
%vecParamsInit = vecParamsTrue; % Override init weights with true weights

% Optimize model
options = optimset('MaxFunEvals',100000,'MaxIter',100000);
vecParams = fminsearch(@(vecParams)(objectiveFunction(x1_train,x2_train,sizeParams,vecParams,row)),vecParamsInit,options);

% Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

H = mlpModel(x1_test,params,row);
MSE=sum((x2_test-H).^2)/size(x2_test,2);
hold on
plot(x1_test,H,'.m')
title(['The test performance with MSE: ',num2str(MSE)])
legend('original','fitted')


%% Functions
function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams,type)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params,type);
objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
%objFncValue = sum(-sum(Y.*log(H),1),2)/N;
% Change objective function to make this MLE for class posterior modeling
end

function H = mlpModel(X,params,type)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U,type);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
end

function out = activationFunction(in,type)
if type==1
    out = log(1+exp(in)); % softplus
else
    out = 1./(1+exp(-in)); % logistic function
    %     out = in./sqrt(1+in.^2); % ISRU
end
end

% function error= Error(yValidate,H)
% Class = H==max(H);      %make max probability 1 and other ones 0
% A=size(yValidate,1);
% B=size(yValidate,2);
% C=0;
% for i=1:A
%     for j=1:B
%         if yValidate(i,j)==1 && Class(i,j)==1
%             C=C+1;
%         end
%         j=j+1;
%     end
%     i=i+1;
% end
% error=1-(C/B);
% end
