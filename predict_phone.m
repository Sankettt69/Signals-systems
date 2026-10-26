%% predict_phone.m - classify activity from ONE phone strapped to the thigh
% mode = 'live' : stream from MATLAB Mobile (needs MATLAB Connector + mobiledev)
% mode = 'file' : analyse a log saved from MATLAB Mobile (variable 'Acceleration')
% IMPORTANT: stand still for the FIRST 5 SECONDS - this calibrates the phone axes
% to the HARTH thigh sensor axes. Strap the phone on the front of the thigh, firmly.
clear; clc; close all;
mode = 'file';   runSeconds = 90;
S = load('harth_thigh_model.mat'); W = S.windowSize; Fs = S.Fs; g = 9.81;

if strcmp(mode,'live')
    connector on; m = mobiledev; m.SampleRate = Fs;
    m.AccelerationSensorEnabled = 1; m.Logging = 1;
    fprintf('Stand still for 5 s (calibration)...\n'); pause(6);
    [a,t] = accellog(m); [a,t] = uniform(a,t,Fs); a = a/g;
    Rm = findAxisMap(median(a(t<=t(1)+4,:),1)', S.standingMean);
    fprintf('Calibrated. Start moving.\n');
    fig = figure('Name','Live phone activity'); ax = axes(fig); tStart = tic;
    while toc(tStart) < runSeconds && isvalid(fig)
        [a,t] = accellog(m); 
        if t(end)-t(1) > W/Fs+0.5
            [a,t] = uniform(a(t>t(end)-(W/Fs+1),:), t(t>t(end)-(W/Fs+1)), Fs);
            win = (Rm*(a(end-W+1:end,:)'/g));
            pr = scores2label(minibatchpredict(S.net, single(harthFeatures(win,Fs))), S.classNames);
            plot(ax,win'); grid(ax,'on'); ylabel(ax,'thigh acc (g)');
            title(ax,sprintf('Detected: %s',string(pr)));
            if pr == "Running", disp('Action: Running detected!'); end
        end
        pause(0.5); drawnow;
    end
    m.Logging = 0; clear m;
else
    [f,p] = uigetfile('*.mat','Select MATLAB Mobile log'); if isequal(f,0), return; end
    L = load(fullfile(p,f)); A = L.Acceleration;
    t = seconds(A.Timestamp - A.Timestamp(1)); a = A{:,:}; 
    [a,t] = uniform(a,t,Fs); a = a/g;
    Rm = findAxisMap(median(a(t<=t(1)+4,:),1)', S.standingMean);
    starts = 1:64:(size(a,1)-W+1); pred = categorical(strings(numel(starts),1),S.classNames);
    for i = 1:numel(starts)
        win = Rm*a(starts(i):starts(i)+W-1,:)';
        pred(i) = scores2label(minibatchpredict(S.net, single(harthFeatures(win,Fs))), S.classNames);
    end
    tt = (starts+W/2)/Fs; figure; plot(tt,double(pred),'.-'); grid on
    yticks(1:numel(S.classNames)); yticklabels(S.classNames); xlabel('Time (s)');
    title('Phone (thigh) - predicted activity'); disp(table(tt(1:min(10,end))',pred(1:min(10,end))));
end

function [a,t] = uniform(a,t,Fs)
    [t,iu] = unique(t); a = a(iu,:); tq = (t(1):1/Fs:t(end))';
    a = interp1(t,a,tq,'linear'); t = tq;
end
function Rm = findAxisMap(phoneStand, harthStand)
% Best signed axis permutation so the phone's standing gravity vector matches HARTH's.
    P = perms(1:3); best = inf; Rm = eye(3);
    for i = 1:size(P,1)
        for s = 0:7
            sg = 1 - 2*bitget(s,1:3); I3 = eye(3); R = diag(sg)*I3(P(i,:),:);
            e = norm(R*phoneStand - harthStand);
            if e < best, best = e; Rm = R; end
        end
    end
end
