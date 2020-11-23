clear all, close all, clc;

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';


Kvalues = 2; % desired numbers of clusters

for imageCounter = 1:2 %size(filenames,2)
    imdata = imread(filenames{1,imageCounter});
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1), imshow(imdata);
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
    %y=length(x);
    [alpha,mu,Sigma] = EMforGMM(Kvalues,x,length(x));
    
    % pdf1 = evalGMM(x,alpha(1),mu(:, 1),Sigma(:, :, 1));
    % pdf2 = evalGMM(x,alpha(2),mu(:, 2),Sigma(:, :, 2
    pdf1 = alpha(1) * evalGaussian(x, mu(:, 1), Sigma(:, :, 1));
    pdf2 = alpha(2) * evalGaussian(x, mu(:, 2), Sigma(:, :, 2));
    
    %MAP
    p_condition = [pdf1;pdf2];
    [~,p] = max(p_condition,[],1);
    
    picture = reshape(p, R, C);
    figure(1), subplot(size(filenames, 2), length(Kvalues) + 1, (imageCounter - 1) * (length(Kvalues) + 1) + 1 + 1), imshow(uint8(picture * 255 / Kvalues));
    title(strcat({'Clustering with K = '}, num2str(2)));
end


%% Function
function [alpha_est,mu,Sigma]=EMforGMM(M,x,N)
delta = 0.3; % tolerance for EM stopping criterion
reg_weight = 1e-2; % regularization parameter for covariance estimates
d = size(x,1); %dimensionality of data

% Initialize the GMM to randomly selected samples
alpha_est= ones(1,M)/M; %start with equal alpha estimates

% Set initial mu as random M value pairs from data array
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M));
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean

for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + reg_weight*eye(d,d);
end
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

%Run EM algorith until it converges
Converged = 0; % Not converged at the beginning
for i=1:10000  %Calculate GMM distribution according to parameters
    for l = 1:M
        temp(l,:) = repmat(alpha_est(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    pl_given_x = temp./sum(temp,1);
    %clear temp
    alpha_new = mean(pl_given_x,2);
    w = pl_given_x./repmat(sum(pl_given_x,2),1,N);
    mu_new = x*w';
    for l = 1:M
        v = x-repmat(mu_new(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        Sigma_new(:,:,l) = u*v' + reg_weight*eye(d,d); % adding a small regularization term
    end
    
    Dalpha = sum(abs(alpha_new-alpha_est'));
    Dmu = sum(sum(abs(mu_new-mu)));
    DSigma = sum(sum(abs(abs(Sigma_new-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    %     if Converged
    %         break
    %     end
    alpha_est = alpha_new; mu = mu_new; Sigma = Sigma_new;
    t=t+1;
    i=i+1;
end
end

function gmm = evalGMM(x,alpha,mu,Sigma)
% Evaluates GMM on the grid based on parameter values given
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end