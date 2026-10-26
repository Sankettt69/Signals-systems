%% predict_harth.m  -  run the trained model on a HARTH CSV or a phone log
clear; clc; close all;
S = load('harth_trained_model.mat');
net = S.net; classNames = S.classNames; W = S.windowSize; step = S.stepSize;
chanNames = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"];

%% Choose source: 'harth' or 'phone'
source = 'harth';

if strcmp(source,'harth')
    [f,p] = uigetfile('*.csv','Select a HARTH subject CSV (e.g. S029.csv)');
    if isequal(f,0), return; end
    T   = readtable(fullfile(p,f),'VariableNamingRule','preserve');
    sig = T{:, cellstr(chanNames(S.useChannels))}';      % [C x N]
    lab = T.label;
else
    % MATLAB Mobile log saved as .mat: expects variable 'Acceleration'
    % (timetable with X,Y,Z in m/s^2). One phone = one sensor, so train
    % with useChannels = 4:6 (thigh only) first.
    [f,p] = uigetfile('*.mat','Select MATLAB Mobile log');
    if isequal(f,0), return; end
    L = load(fullfile(p,f));
    A = L.Acceleration;
    A = retime(A, 'regular','linear','SampleRate',S.Fs);
    sig = (A{:,{'X','Y','Z'}} / 9.81)';                  % convert to g
    lab = [];
    if numel(S.useChannels) ~= 3
        error('Phone data has one sensor: retrain with useChannels = 4:6.');
    end
end

%% Sliding-window prediction
starts = 1:step:(size(sig,2)-W+1);
Xw = arrayfun(@(s) single(sig(:,s:s+W-1))', starts, 'UniformOutput', false)';
scores = minibatchpredict(net, Xw);
pred   = scores2label(scores, classNames);

%% Report
if ~isempty(lab)
    rl = arrayfun(@(s) mode(lab(s:s+W-1)), starts)';
    [tf,k] = ismember(rl, S.rawLabels);
    truth = categorical(strings(numel(rl),1), classNames); 
    truth(tf) = classNames(k(tf));
    keep = tf;
    fprintf('Window accuracy on %s: %.2f%% (%d labeled windows)\n', ...
        f, 100*mean(pred(keep)==truth(keep)), nnz(keep));
    figure; confusionchart(truth(keep), pred(keep),'Normalization','row-normalized');
end
t = (starts + W/2)/S.Fs;
figure; plot(t, double(pred),'.-'); yticks(1:numel(classNames)); yticklabels(classNames);
xlabel('Time (s)'); title('Predicted activity over time'); grid on;
disp(table(t(1:min(10,end))', pred(1:min(10,end)), 'VariableNames',{'Time_s','Activity'}));
