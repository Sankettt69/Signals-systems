%% predict_pretrained.m - evaluate the friend's pre-trained model on a HARTH CSV
clear; clc; close all;
kind = 'mat';     % 'mat' (CNN+BiLSTM) | 'onnx' | 'onnx_corrected'
M = loadHarthModel(kind);
chan = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"];
rawLabels = [1 2 3 4 5 6 7 8 13 14 130 140];
fileNames = ["Walking","Running","Shuffling","Stairs (ascending)","Stairs (descending)", ...
    "Standing","Sitting","Lying","Cycling (sit)","Cycling (stand)", ...
    "Cycling (sit, inactive)","Cycling (stand, inactive)"];

[f,p] = uigetfile('*.csv','Select a HARTH CSV (e.g. S029.csv)'); if isequal(f,0), return; end
T = readtable(fullfile(p,f),'VariableNamingRule','preserve');
sig = T{:,cellstr(chan)}'; lab = T.label; W = M.windowSize;
starts = 1:M.stepSize:(size(sig,2)-W+1);
Xc = arrayfun(@(s) single(sig(:,s:s+W-1)), starts, 'UniformOutput', false)';
pred = M.predict(Xc);

rl = arrayfun(@(s) mode(lab(s:s+W-1)), starts)';
[tf,k] = ismember(rl, rawLabels);
truth = categorical(fileNames(max(k,1))', M.classNames);
acc = mean(pred(tf)==truth(tf));
fprintf('\n%s model on %s: window accuracy %.2f%% (%d labeled windows)\n', kind, f, 100*acc, nnz(tf));
figure; confusionchart(truth(tf), pred(tf), 'Normalization','row-normalized', ...
    'Title', sprintf('%s on %s: %.1f%%', kind, f, 100*acc));
t = (starts+W/2)/M.Fs;
figure; plot(t, double(pred),'.', t(tf), double(truth(tf)),'-'); grid on
yticks(1:numel(M.classNames)); yticklabels(M.classNames); xlabel('Time (s)');
legend('Predicted','True'); title('Activity over time');
