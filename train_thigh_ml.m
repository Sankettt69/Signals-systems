%% train_thigh_ml.m - thigh-only model, higher-accuracy version (for ONE phone)
% Needs only Deep Learning Toolbox. Merges look-alike classes and uses overlap.
clear; clc; close all; rng(0);
mergeStairs = true;                 % false = keep ascending/descending separate
Fs = 50; windowSize = 128; stepSize = 64;
rawLabels = [1 2 3 4 5 6 7 8 13 14 130 140];
if mergeStairs
    rawNames = ["Walking","Running","Shuffling","Stairs","Stairs","Standing","Sitting","Lying", ...
                "Cycling","Cycling","Cycling","Cycling"];
else
    rawNames = ["Walking","Running","Shuffling","Stairs (ascending)","Stairs (descending)", ...
                "Standing","Sitting","Lying","Cycling","Cycling","Cycling","Cycling"];
end
classNames = unique(rawNames,'stable');
chan = {'thigh_x','thigh_y','thigh_z'};

cands = {fullfile('data','harth'),'harth dataset',fullfile('dataset','harth'),'harth','.'};
dataDir = '';
for c = 1:numel(cands)
    if ~isempty(dir(fullfile(cands{c},'S*.csv'))), dataDir = cands{c}; break; end
end
if isempty(dataDir), error('No S*.csv HARTH files found.'); end
files = dir(fullfile(dataDir,'S*.csv'));

F = []; Y = strings(0,1); subj = []; standMeans = [];
for i = 1:numel(files)
    T = readtable(fullfile(files(i).folder,files(i).name),'VariableNamingRule','preserve');
    sig = T{:,chan}'; lab = T.label;
    for s = 1:stepSize:(size(sig,2)-windowSize+1)
        idx = s:s+windowSize-1; k = find(rawLabels==mode(lab(idx)),1);
        if isempty(k) || any(~isfinite(sig(:,idx)),'all'), continue; end
        F(end+1,:) = harthFeatures(sig(:,idx),Fs); %#ok<SAGROW>
        Y(end+1,1) = rawNames(k);                  %#ok<SAGROW>
        subj(end+1,1) = i;                         %#ok<SAGROW>
        if rawNames(k) == "Standing", standMeans(end+1,:) = mean(sig(:,idx),2)'; end %#ok<SAGROW>
    end
    fprintf('%s done (%d/%d)\n',files(i).name,i,numel(files));
end
F = single(F); Y = categorical(Y,classNames);
standingMean = median(standMeans,1)';

ids = unique(subj); ids = ids(randperm(numel(ids)));
nTe = max(1,round(0.2*numel(ids))); nVa = max(1,round(0.15*numel(ids)));
isTe = ismember(subj, ids(1:nTe)); isVa = ismember(subj, ids(nTe+1:nTe+nVa));
isTr = ~(isTe|isVa);
fprintf('Train %d / Val %d / Test %d windows, %d classes\n',nnz(isTr),nnz(isVa),nnz(isTe),numel(classNames));

counts = countcats(Y(isTr)); w = min(sum(counts)./(numel(counts)*max(counts,1)), 10);
lossFcn = @(Yp,T) crossentropy(Yp,T,single(w(:)),'WeightsFormat','C','NormalizationFactor','batch-size');
layers = [
    featureInputLayer(size(F,2),'Normalization','zscore')
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numel(classNames))
    softmaxLayer];
opts = trainingOptions('adam','MaxEpochs',60,'MiniBatchSize',256,'InitialLearnRate',2e-3, ...
    'LearnRateSchedule','piecewise','LearnRateDropPeriod',20,'LearnRateDropFactor',0.3, ...
    'Shuffle','every-epoch','ValidationData',{F(isVa,:),Y(isVa)},'ValidationFrequency',50, ...
    'ValidationPatience',8,'OutputNetwork','best-validation', ...
    'Plots','training-progress','Verbose',false);
net = trainnet(F(isTr,:),Y(isTr),layers,lossFcn,opts);

YPred = scores2label(minibatchpredict(net,F(isTe,:)),classNames);
acc = mean(YPred==Y(isTe));
fprintf('Thigh-only test accuracy (unseen subjects): %.2f%%\n',100*acc);
figure; confusionchart(Y(isTe),YPred,'Normalization','row-normalized', ...
    'Title',sprintf('Thigh-only model %.1f%%',100*acc));
save('harth_thigh_model.mat','net','classNames','windowSize','Fs','standingMean');
fprintf('Saved harth_thigh_model.mat\n');
