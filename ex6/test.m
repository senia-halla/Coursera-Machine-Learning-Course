% Part 1
clc ; 
clear all ; 
close all ; 


% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
figure(1);
plotData(X, y);

% Good Value of C 
figure(2);
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure(2);
visualizeBoundaryLinear(X, y, model);

% Small value of C : Underfitt
figure(3);
C = 0.01;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure(2);
visualizeBoundaryLinear(X, y, model);

% Big value of C : Overfitt
% you should find that the SVM now classifies every single example correctly, but has a decision boundary 
%that does not appear to be a natural fit for the data 
figure(4);
C = 10000;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure(2);
visualizeBoundaryLinear(X, y, model);

x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);


% Part 2 : 
clc ; 
clear all ; 
close all ; 

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);  


% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

% Part 3 : 
clc ;
clear all ;
close all ;

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);


%% Initialization
clear;

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
% Print Stats
disp(word_indices)

% Extract Features
features = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

%%
clear ;
clc ; 
% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);