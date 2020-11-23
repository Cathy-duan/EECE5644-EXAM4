clear all, close all, clc;

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';

for imageCounter = 1:2 %size(filenames,2)
    imdata = imread(filenames{1,imageCounter});
    figure(1), subplot(size(filenames,2),1+1,(imageCounter-1)*(1+1)+1), imshow(imdata);
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);% overwriting, since I don't need the uint8 format anymore
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        color = imdata(:,:,d); % pick one color at a time
        features = [features;color(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    
    d = size(x,1); % feature dimensionality
    
    %GMM based clustering
    
    Kvalues = cross_validation(x,length(x));
    
    GMModel=fitgmdist(x',Kvalues,'RegularizationValue',1e-5);
    alpha = GMModel.ComponentProportion;
    mu = (GMModel.mu)';
    sigma = GMModel.Sigma;
    
    %make pdf
    for a = 1:Kvalues
        pdf(a,:) = alpha(a) * evalGaussian(x, mu(:, a), sigma(:, :, a));
    end
    
    %MAP
    [~,p] = max(pdf,[],1);
    picture = reshape(p, R, C);
    figure(1), subplot(size(filenames, 2), length(Kvalues) + 1, (imageCounter - 1) * (length(Kvalues) + 1) + 1 + 1), imshow(uint8(picture * 255 / Kvalues));
    title(strcat({'Clustering with K = '}, num2str(Kvalues)));
end

%% number of Gaussian components
function best_clusters = cross_validation(X,N)

% Performs EM algorithm to estimate parameters and evaluete performance
% on each data set B times, with 1 through M GMM models considered

K = 10; M = 10;            %repetitions per data set; max GMM considered
perf_array = zeros(K,M);    %save space for performance evaluation


dummy = ceil(linspace(0,N,K+1));       % Divide the data set into K approximately-equal-sized partitions
for n = 1:K
    indPartitionLimits(n,:) = [dummy(n)+1,dummy(n+1)];
end


for k = 1:K                     % Parse data into K folds....
    
    indValidate = (indPartitionLimits(k, 1) : indPartitionLimits(k, 2));
    if k == 1
        indTrain = [indPartitionLimits(k,2)+1:N];
    elseif k == K
        indTrain = [1:indPartitionLimits(k,1)-1];
    else
        indTrain = [1:indPartitionLimits(k-1,2) indPartitionLimits(k+1,1):N];
    end
    
    xTrain = X(:,indTrain);       % Using all other folds as training set
    
    xValidate = X(:,indValidate); % Using fold k as validation set
    
    for m=1:M
        
        
        %Non-Buil-In:run EM algorith to estimate parameters
        %[alpha,mu,sigma]=EMforGMM(m,xTrain,size(xTrain,2),xValidate);
        
        %Built-In function: run EM algorithm to estimate parameters
        GMModel=fitgmdist(xTrain',m,'RegularizationValue',1e-10);
        alpha = GMModel.ComponentProportion;
        mu = (GMModel.mu)';
        sigma = GMModel.Sigma;
        
        % Calculate log-likelihood performance with new parameters
        perf_array(k,m)=sum(log(evalGMM(xValidate,alpha,mu,sigma)));
    end
end
% Calculate average performance for each M and find best fit

avg_perf =sum(perf_array)/K;
disp(avg_perf);
best_clusters = find(avg_perf == max(avg_perf),1);
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
% Evaluates GMM on the grid based on parameter values given
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end