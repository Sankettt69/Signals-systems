%% train_harth.m  -  HARTH activity recognition (1D-CNN + BiLSTM), pure MATLAB
% Requires: Deep Learning Toolbox (R2023b or newer, uses trainnet)
% Run from the folder that contains the HARTH CSVs (or a subfolder named
% 'harth dataset', 'dataset/harth' or 'harth').
clear; clc; close all; rng(0);

%% Settings
useChannels = 1:6;      % 1:6 = back+thigh. Use 4:6 for thigh only (single phone/IMU)
Fs          = 50;       % HARTH sampling rate (Hz)
windowSize  = 128;      % ~2.56 s
stepSize    = 64;       % 50% overlap
rawLabels   = [1 2 3 4 5 6 7 8 13 14 130 140];
classNames  = ["Walking","Running","Shuffling","Stairs (ascending)", ...
               "Stairs (descending)","Standing","Sitting","Lying", ...
               "Cycling (sit)","Cycling (stand)", ...
               "Cycling (sit, inactive)","Cycling (stand, inactive)"];
chanNames   = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"];

%% Locate data
cands = {fullfile('data','harth'), 'harth dataset', fullfile('dataset','harth'), 'harth', '.'};
dataDir = '';
for c = 1:numel(cands)
    if ~isempty(dir(fullfile(cands{c},'S*.csv'))), dataDir = cands{c}; break; end
end
if isempty(dataDir), error('No S*.csv HARTH files found. cd to the dataset folder.'); end
files = dir(fullfile(dataDir,'S*.csv'));
fprintf('Found %d subject files in "%s"\n', numel(files), dataDir);

%% Windowing (keeps subject id so we can split BY SUBJECT)
X = {}; Y = {}; subj = [];
for i = 1:numel(files)
    T = readtable(fullfile(files(i).folder, files(i).name), ...
        'VariableNamingRule','preserve');
    sig = T{:, cellstr(chanNames(useChannels))}';      % [C x N]
    lab = T.label;
    for s = 1:stepSize:(size(sig,2) - windowSize + 1)
        idx = s:s+windowSize-1;
        m = mode(lab(idx));
        k = find(rawLabels == m, 1);
        if isempty(k) || any(~isfinite(sig(:,idx)),'all'), continue; end
        X{end+1,1} = single(sig(:,idx))';   % [T x C]  %#ok<SAGROW>
        Y{end+1,1} = classNames(k);       %#ok<SAGROW>
        subj(end+1,1) = i;                %#ok<SAGROW>
    end
    fprintf('  %s done\n', files(i).name);
end
Y = categorical(string(Y), classNames);   % fixed class order = classNames
fprintf('Total windows: %d\n', numel(X));

%% Subject-independent split (avoids leakage from overlapping windows)
ids  = unique(subj);
ids  = ids(randperm(numel(ids)));
nTe  = max(1, round(0.2*numel(ids)));
nVa  = max(1, round(0.15*numel(ids)));
teS  = ids(1:nTe);  vaS = ids(nTe+1:nTe+nVa);
isTe = ismember(subj,teS); isVa = ismember(subj,vaS); isTr = ~(isTe|isVa);
XTrain = X(isTr); YTrain = Y(isTr);
XVal   = X(isVa); YVal   = Y(isVa);
XTest  = X(isTe); YTest  = Y(isTe);
fprintf('Train/Val/Test windows: %d / %d / %d\n', nnz(isTr), nnz(isVa), nnz(isTe));

%% Class weights (HARTH is heavily imbalanced)
counts  = countcats(YTrain);
weights = sum(counts) ./ (numel(counts) * max(counts,1));
weights = min(weights, 20);
lossFcn = @(Yp,T) crossentropy(Yp, T, single(weights(:)), ...
    'WeightsFormat','C','NormalizationFactor','batch-size');

%% Network
numCh = numel(useChannels); numClasses = numel(classNames);
layers = [
    sequenceInputLayer(numCh,'Normalization','zscore','Name','in')
    convolution1dLayer(5,64,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling1dLayer(2,'Stride',2,'Padding','same','Name','pool1')
    convolution1dLayer(5,128,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling1dLayer(2,'Stride',2,'Padding','same','Name','pool2')
    bilstmLayer(100,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.4,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')];

opts = trainingOptions('adam', ...
    'MaxEpochs',15,'MiniBatchSize',128,'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise','LearnRateDropPeriod',8,'LearnRateDropFactor',0.2, ...
    'Shuffle','every-epoch','ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',50,'Plots','training-progress','Verbose',false);

%% Train
net = trainnet(XTrain, YTrain, layers, lossFcn, opts);

%% Evaluate on unseen subjects
scores = minibatchpredict(net, XTest);
YPred  = scores2label(scores, classNames);
acc = mean(YPred == YTest);
fprintf('\nTest accuracy (unseen subjects): %.2f%%\n', 100*acc);
figure('Name','Confusion matrix');
confusionchart(YTest, YPred, 'Normalization','row-normalized', ...
    'Title', sprintf('HARTH test accuracy %.1f%%',100*acc));

save('harth_trained_model.mat','net','classNames','rawLabels', ...
     'windowSize','stepSize','Fs','useChannels');
fprintf('Saved harth_trained_model.mat\n');
