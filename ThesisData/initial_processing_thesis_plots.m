clear all

ps = readtable('ps_calib_SOL0239.csv');

[sol,temp] = strtok(ps.LTST,' ');

pressure_rate = mean(ps.PRESSURE_FREQUENCY);

for i=1:length(temp)
    time = temp{i}(2:end);
    temp2 = split(time,':');
    hh = str2double(temp2{1});
    mm = str2double(temp2{2});
    ss = str2double(temp2{3});
    time_s = (hh*3600) + (mm * 60) + ss;

    decimal_sol(i) = str2double(sol{i}) + (time_s/86400);
end

one_sec_in_sol = 1/86400;

% Find groups of identical integer-second sols
[G,~,groupIdx] = unique(decimal_sol,'stable');
decimal_sol_frac = decimal_sol;

for k = 1:numel(G)
    idx   = find(groupIdx==k);
    count = numel(idx);

    if count==1
        offsets = 0;
    elseif k==1 && count<pressure_rate
        % first partial group: place at end of second
        start_frac = 1 - count/pressure_rate;
        offsets = (start_frac + (0:count-1)/pressure_rate);
    elseif k==numel(G) && count<pressure_rate
        % last partial group: place at start of second
        offsets = (0:count-1)/pressure_rate;
    else
        % full or middle groups
        offsets = (0:count-1)/pressure_rate;
    end

    decimal_sol_frac(idx) = decimal_sol_frac(idx) + offsets*one_sec_in_sol;
end

decimal_sol = decimal_sol_frac;
%%
figure
tiledlayout(3,1)
nexttile(1)
plot(decimal_sol,ps.PRESSURE)
ylabel('$P$ [Pa]')
xlim([239,240])
set(gca, 'XTickLabel', [])

[lowpassed_pressure,~] = lowpass_2Hz(ps.PRESSURE,pressure_rate);

window_size = pressure_rate * 1000;

detrended_pressure = NaN(length(lowpassed_pressure),1);

background = NaN(length(lowpassed_pressure),1);


for i=1:1
        for j = 1:length(lowpassed_pressure)-window_size
            window_values = lowpassed_pressure(j : j + window_size, i);
            window_mean = mean(window_values);
            detrended_pressure(j + floor(window_size/ 2), i) = lowpassed_pressure(j + floor(window_size / 2), i) - window_mean;
            background(j + floor(window_size / 2), i) = window_mean;
        end
end

nexttile(2)
plot(decimal_sol,detrended_pressure)
ylabel('$\Delta{}P$ [Pa]')
set(gca, 'XTickLabel', [])
xlim([239,240])

nexttile(3)
plot(decimal_sol,background)
ylabel('$P_{BGR}$ [Pa]')
xlabel('Decimal Sol [-]')
xlim([239,240])

%%
cutoff_frequency = 2.0;  % Hz
nyquist_frequency = pressure_rate / 2.0;  % Nyquist frequency
normalized_cutoff = cutoff_frequency / nyquist_frequency;
[b, a] = butter(2, normalized_cutoff, 'low');
figure
% Plot the frequency response
freqz(b, a, 1024, pressure_rate);
title('Frequency Response of the 2 Hz Low-pass Butterworth Filter');
%%
[~,pressure_peaks] = findpeaks(-detrended_pressure,"MinPeakHeight",0.35,"MinPeakDistance",50 * pressure_rate);

figure
plot(decimal_sol,detrended_pressure)
hold on
for i = 1:length(pressure_peaks)
    plot(repmat(decimal_sol(pressure_peaks(i)),2),[-5,1],'k--')
end
xlabel('Decimcal Sol [-]')
ylabel('$\Delta{}P$ [Pa]')

[p_temp(:,1), p_temp(:,2), p_temp(:,3)] = hms(seconds((decimal_sol(pressure_peaks) - 239) * 86400));
%%
window_duration = 200;
p_windows = [pressure_peaks-(window_duration/2 * pressure_rate),pressure_peaks+(window_duration/2 * pressure_rate)];
for ii = 1:length(pressure_peaks)
    threshold = detrended_pressure(pressure_peaks(ii))/2;
    values = detrended_pressure;
    peak = pressure_peaks(ii);
    for i = peak:-1:p_windows(ii,1)
            if values(i) <= threshold && values(i-1) >= threshold
                p_walls(ii,1) = i;
                break
            end
    end
    for i = peak:p_windows(ii,2)
        if values(i) <= threshold && values(i+1) >= threshold
            p_walls(ii,2) = i;
            break
        end
    end
    
end

ii = 1;

figure
plot(decimal_sol(p_windows(ii,1):p_windows(ii,2)),detrended_pressure(p_windows(ii,1):p_windows(ii,2)))
hold on
plot([decimal_sol(pressure_peaks(ii)) decimal_sol(pressure_peaks(ii))],[-1,1],'k--')
plot([decimal_sol(p_walls(ii,1)) decimal_sol(p_walls(ii,1)) NaN decimal_sol(p_walls(ii,2)) decimal_sol(p_walls(ii,2))],[-1,1,NaN,-1,1],'m:')
xlabel('Decimal Sol [-]')
ylabel('$\Delta{})$ [Pa]')

%%
t_FWHM = decimal_sol(p_walls(:,2)) - decimal_sol(p_walls(:,1));

colours = lines(4);

ii = 1;
figure
l1 = plot(decimal_sol(p_windows(ii,1):p_windows(ii,2)),detrended_pressure(p_windows(ii,1):p_windows(ii,2)));
hold on

baseline_lor = lv_pressure(decimal_sol(p_windows(ii,1):p_windows(ii,2)),0,detrended_pressure(pressure_peaks(ii)),decimal_sol(pressure_peaks(ii)),0,t_FWHM(ii));
l2 = plot(decimal_sol(p_windows(ii,1):p_windows(ii,2)),baseline_lor,'Color',colours(2,:));
plot([decimal_sol(p_windows(ii,1)) decimal_sol(pressure_peaks(ii))],[detrended_pressure(pressure_peaks(ii)),detrended_pressure(pressure_peaks(ii))],'--','Color',colours(2,:))

display(mean((detrended_pressure(p_windows(ii,1):p_windows(ii,2))-baseline_lor').^2));

lorentz_fixed = @(params,x) -abs(params(1)) ./ (1 + ((2*(x - params(2))/t_FWHM(ii)).^2));
params0 = [detrended_pressure(pressure_peaks(ii)), decimal_sol(pressure_peaks(ii))];

params_fit = lsqcurvefit(lorentz_fixed, params0, decimal_sol(p_windows(ii,1):p_windows(ii,2))', detrended_pressure(p_windows(ii,1):p_windows(ii,2)));
fixed_D_lsq_lor = lv_pressure(decimal_sol(p_windows(ii,1):p_windows(ii,2)),0,params_fit(1),params_fit(2),0,t_FWHM(ii));
l3 = plot(decimal_sol(p_windows(ii,1):p_windows(ii,2)),fixed_D_lsq_lor,'Color',colours(3,:));
plot([decimal_sol(p_windows(ii,1)) params_fit(2)],[params_fit(1),params_fit(1)],'--','Color',colours(3,:))

display(mean((detrended_pressure(p_windows(ii,1):p_windows(ii,2))-fixed_D_lsq_lor').^2));


lorentz_free = @(params,x) -abs(params(1)) ./ (1 + ((2*(x - params(2))/params(3)).^2));
params0 = [detrended_pressure(pressure_peaks(ii)), decimal_sol(pressure_peaks(ii)),t_FWHM(ii)];

params_fit = lsqcurvefit(lorentz_free, params0, decimal_sol(p_windows(ii,1):p_windows(ii,2))', detrended_pressure(p_windows(ii,1):p_windows(ii,2)));
free_D_lsq_lor = lv_pressure(decimal_sol(p_windows(ii,1):p_windows(ii,2)),0,params_fit(1),params_fit(2),0,t_FWHM(ii));
l4 = plot(decimal_sol(p_windows(ii,1):p_windows(ii,2)),free_D_lsq_lor,'Color',colours(4,:));
plot([decimal_sol(p_windows(ii,1)) params_fit(2)],[params_fit(1),params_fit(1)],'--','Color',colours(4,:))

display(mean((detrended_pressure(p_windows(ii,1):p_windows(ii,2))-free_D_lsq_lor').^2));

legend([l1,l2,l3,l4],'$\Delta{}P$ Trace','Pressure Trace Lorentzian','Fixed FWHM LS Lorentzian','Flexible FWHM LS Lorentzian')

xlabel('Decimal Sol [-]')
ylabel('$\Delta{})$ [Pa]')


%% TEMPERATURE
twins = readtable('twins_model_SOL0239.csv');

temperature_rate = mean(twins.BPY_AIR_TEMP_FREQUENCY);

[sol,temp] = strtok(twins.LTST,' ');


for i=1:length(temp)
    time = temp{i}(2:end);
    temp2 = split(time,':');
    hh = str2double(temp2{1});
    mm = str2double(temp2{2});
    ss = str2double(temp2{3});
    time_s = (hh*3600) + (mm * 60) + ss;

    decimal_sol_t(i) = str2double(sol{i}) + (time_s/86400);
end

% Find groups of identical integer-second sols
[G,~,groupIdx] = unique(decimal_sol_t,'stable');
decimal_sol_frac_t = decimal_sol_t;

for k = 1:numel(G)
    idx   = find(groupIdx==k);
    count = numel(idx);

    if count==1
        offsets = 0;
    elseif k==1 && count<temperature_rate
        % first partial group: place at end of second
        start_frac = 1 - count/temperature_rate;
        offsets = (start_frac + (0:count-1)/temperature_rate);
    elseif k==numel(G) && count<temperature_rate
        % last partial group: place at start of second
        offsets = (0:count-1)/temperature_rate;
    else
        % full or middle groups
        offsets = (0:count-1)/temperature_rate;
    end

    decimal_sol_frac_t(idx) = decimal_sol_frac_t(idx) + offsets*one_sec_in_sol;
end

decimal_sol_t = decimal_sol_frac_t;

%%
figure
tiledlayout(3,1)
nexttile(1)
plot(decimal_sol_t,twins.BMY_AIR_TEMP)
ylabel('$T$ [K]')
xlim([239,240])
set(gca, 'XTickLabel', [])

window_size = temperature_rate * 200;

temperature = twins.BMY_AIR_TEMP;

detrended_temperature = NaN(length(temperature),1);
background = NaN(length(temperature),1);

for i=1:1
        for j = 1:length(temperature)-window_size
            window_values = temperature(j : j + window_size, i);
            window_mean = mean(window_values);
            detrended_temperature(j + floor(window_size/ 2), i) = temperature(j + floor(window_size / 2), i) - window_mean;
            background(j + floor(window_size / 2), i) = window_mean;
        end
end

nexttile(2)
plot(decimal_sol_t,detrended_temperature)
ylabel('$\Delta{}T$ [K]')
xlim([239,240])
set(gca, 'XTickLabel', [])

nexttile(3)
plot(decimal_sol_t,background)
ylabel('$T_{BGR}$ [K]')
xlim([239,240])
xlabel('Decimal Sol [-]')
%% WINDSPEED
twins = readtable('twins_model_SOL0239.csv');

[sol,temp] = strtok(twins.LTST,' ');

for i=1:length(temp)
    time = temp{i}(2:end);
    temp2 = split(time,':');
    hh = str2double(temp2{1});
    mm = str2double(temp2{2});
    ss = str2double(temp2{3});
    time_s = (hh*3600) + (mm * 60) + ss;

    decimal_sol_t(i) = str2double(sol{i}) + (time_s/86400);
end

twins_rate = mean(twins.WIND_FREQUENCY);

[G,~,groupIdx] = unique(decimal_sol_t,'stable');
decimal_sol_frac_t = decimal_sol_t;

for k = 1:numel(G)
    idx   = find(groupIdx==k);
    count = numel(idx);

    if count==1
        offsets = 0;
    elseif k==1 && count<twins_rate
        % first partial group: place at end of second
        start_frac = 1 - count/twins_rate;
        offsets = (start_frac + (0:count-1)/twins_rate);
    elseif k==numel(G) && count<twins_rate
        % last partial group: place at start of second
        offsets = (0:count-1)/twins_rate;
    else
        % full or middle groups
        offsets = (0:count-1)/twins_rate;
    end

    decimal_sol_frac_t(idx) = decimal_sol_frac_t(idx) + offsets*one_sec_in_sol;
end

decimal_sol_t = decimal_sol_frac_t;

%%
figure
tiledlayout(3,1)
nexttile(1)
plot(decimal_sol_t,twins.HORIZONTAL_WIND_SPEED)
ylabel('$V$ [ms\textsuperscript{-1}]')
xlim([239,240])
set(gca, 'XTickLabel', [])

window_size = twins_rate * 200;

twins_horizontal = twins.HORIZONTAL_WIND_SPEED;

detrended_twins = NaN(length(twins_horizontal),1);
background = NaN(length(twins_horizontal),1);

for i=1:1
        for j = 1:length(twins_horizontal)-window_size
            window_values = twins_horizontal(j : j + window_size, i);
            window_mean = mean(window_values);
            detrended_twins(j + floor(window_size/ 2), i) = twins_horizontal(j + floor(window_size / 2), i) - window_mean;
            background(j + floor(window_size / 2), i) = window_mean;
        end
end

nexttile(2)
plot(decimal_sol_t,detrended_twins)
ylabel('$\Delta{}V$ [ms\textsuperscript{-1}]')
xlim([239,240])
set(gca, 'XTickLabel', [])

nexttile(3)
plot(decimal_sol_t,background)
ylabel('$V_{BGR}$ [ms\textsuperscript{-1}]')
xlim([239,240])
xlabel('Decimal Sol [-]')

%%
twins_windows = NaN(size(p_windows));
    for i = 1:length(pressure_peaks)
        [~,idx1] = min(abs(decimal_sol_t - decimal_sol(p_windows(i,1))));
        twins_windows(i,1) = idx1;
        [~,idx2] = min(abs(decimal_sol_t - decimal_sol(p_windows(i,2))));
        twins_windows(i,2) = idx2;
    end

ii = 1;

figure
tiledlayout(3,1)
nexttile(1)
plot(decimal_sol_t(twins_windows(ii,1):twins_windows(ii,2)),detrended_twins(twins_windows(ii,1):twins_windows(ii,2)))
hold on
plot([params_fit(2) params_fit(2)],[-5,6],'k--');
ylabel('$\Delta{}V$ [ms\textsuperscript{-1}]')
set(gca, 'XTickLabel', [])

nexttile(2)
plot(decimal_sol_t(twins_windows(ii,1):twins_windows(ii,2)),background(twins_windows(ii,1):twins_windows(ii,2)))
hold on
plot([decimal_sol_t(twins_windows(ii,1)) decimal_sol_t(twins_windows(ii,2))],[mean(background(twins_windows(ii,1):twins_windows(ii,2))) mean(background(twins_windows(ii,1):twins_windows(ii,2)))])
plot([params_fit(2) params_fit(2)],[7.4,8.2],'k--');

ylabel('$V_{BGR}$ [ms\textsuperscript{-1}]')
set(gca, 'XTickLabel', [])

nexttile(3)
plot(decimal_sol_t(twins_windows(ii,1):twins_windows(ii,2)),detrended_temperature(twins_windows(ii,1):twins_windows(ii,2)))
hold on
plot([params_fit(2) params_fit(2)],[-1,1],'k--');

ylabel('$\Delta{}T$ [K]')
xlabel('Decimal Sol [-]')

%%

load("FIR_coeffs_5.mat")

Fs = 20;
[H,f] = freqz(b,1,2048,Fs);
figure
plot(f,20*log10(abs(H)))

xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')

%% SEIS
seis = readtable('seis_cal_SOL0239_20Hz_v2.csv');

[sol,temp] = strtok(seis.LTST,' ');

for i=1:length(temp)
    time = temp{i}(2:end);
    temp2 = split(time,':');
    hh = str2double(temp2{1});
    mm = str2double(temp2{2});
    ss = str2double(temp2{3});
    time_s = (hh*3600) + (mm * 60) + ss;

    decimal_sol_s(i) = str2double(sol{i}) + (time_s/86400);
end

seis_rate = 20;

[G,~,groupIdx] = unique(decimal_sol_s,'stable');
decimal_sol_frac_s = decimal_sol_s;

for k = 1:numel(G)
    idx   = find(groupIdx==k);
    count = numel(idx);

    if count==1
        offsets = 0;
    elseif k==1 && count<seis_rate
        % first partial group: place at end of second
        start_frac = 1 - count/seis_rate;
        offsets = (start_frac + (0:count-1)/seis_rate);
    elseif k==numel(G) && count<seis_rate
        % last partial group: place at start of second
        offsets = (0:count-1)/seis_rate;
    else
        % full or middle groups
        offsets = (0:count-1)/seis_rate;
    end

    decimal_sol_frac_s(idx) = decimal_sol_frac_s(idx) + offsets*one_sec_in_sol;
end

decimal_sol_s = decimal_sol_frac_s;


load("FIR_coeffs_5.mat")

FIR_VBB_N = filter(b, 1, seis.VBB_N);
FIR_VBB_E = filter(b, 1, seis.VBB_E);
FIR_VBB_Z = filter(b, 1, seis.VBB_Z);

Wn = [0.02, 0.3]/(seis_rate/2);

[d,c] = butter(4,Wn,"bandpass");

BP_VBB_N = filtfilt(d,c,FIR_VBB_N);
BP_VBB_E = filtfilt(d,c,FIR_VBB_E);
BP_VBB_Z = filtfilt(d,c,FIR_VBB_Z);

dt = 1/seis_rate;

a_VBB_N = gradient(BP_VBB_N, dt);
a_VBB_E = gradient(BP_VBB_E, dt);
a_VBB_Z = gradient(BP_VBB_Z, dt);

%figure
%tiledlayout(3,1)
%nexttile(1)
%plot(decimal_sol_s,seis.VBB_N)
%hold on
%plot(decimal_sol_s,seis.VBB_E)
%plot(decimal_sol_s,seis.VBB_Z)
%ylabel('$V_{seis}$ [ms\textsuperscript{-1}]')
%set(gca, 'XTickLabel', [])
%
%nexttile(2)
%plot(decimal_sol_s,BP_VBB_N)
%hold on
%plot(decimal_sol_s,BP_VBB_E)
%plot(decimal_sol_s,BP_VBB_Z)
%ylabel('$V_{seis}$ [ms\textsuperscript{-1}]')
%set(gca, 'XTickLabel', [])
%
%nexttile(3)
%plot(decimal_sol_s,a_VBB_N)
%hold on
%plot(decimal_sol_s,a_VBB_E)
%plot(decimal_sol_s,a_VBB_Z)
%ylabel('$a_{seis}$ [ms\textsuperscript{-2}]')
%%
figure
tiledlayout(3,1,"TileSpacing","tight")
ax1 = nexttile(1);
set(gca,'xticklabel',[])
ax2 = nexttile(2);
set(gca,'xticklabel',[])
ax3 = nexttile(3);

pos1 = ax1.Position;
pos2 = ax2.Position;
pos3 = ax3.Position;
delete(ax1)
delete(ax2)
delete(ax3)

colours = lines(3);

axes('Position',[pos1(1), pos1(2)+2*pos1(4)/3, pos1(3), pos1(4)/3]);
axN = plot(decimal_sol_s, seis.VBB_Z);
set(gca,'xticklabel',[])

ymin = min(seis.VBB_Z);
ymax = max(seis.VBB_Z);
mag = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag)),floor(ymax/mag));
mult2 = max(abs(floor(ymin/mag)),ceil(ymax/mag));
set(gca,'xticklabel',[])

ylim([-mult2 * mag, mult2 * mag]);
yticks(gca, [-mult * mag, 0, mult *  mag]);xlim(gca,[239,240])
xlim(gca,[239,240])

axes('Position',[pos1(1), pos1(2)+pos1(4)/3, pos1(3), pos1(4)/3]);
axE = plot(decimal_sol_s, seis.VBB_N./mag);
set(gca,'xticklabel',[])
ylabel('$V_{seis}$ [ms\textsuperscript{-1}]')

ymin = min(seis.VBB_N./mag);
ymax = max(seis.VBB_N./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])

axes('Position',[pos1(1), pos1(2), pos1(3), pos1(4)/3]);
axZ = plot(decimal_sol_s, seis.VBB_E./mag);
set(gca,'xticklabel',[])

ymin = min(seis.VBB_E./mag);
ymax = max(seis.VBB_E./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])

set(axN, 'Color', colours(1,:));
set(axE, 'Color', colours(2,:));
set(axZ, 'Color', colours(3,:));

%----------------------------------------------
axes('Position',[pos2(1), pos2(2)+2*pos2(4)/3, pos2(3), pos2(4)/3]);
axZ = plot(decimal_sol_s, BP_VBB_Z);
set(gca,'xticklabel',[])

ymin = min(BP_VBB_Z);
ymax = max(BP_VBB_Z);
mag = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag)),floor(ymax/mag));
mult2 = max(abs(floor(ymin/mag)),ceil(ymax/mag));
ylim([-mult2 * mag, mult2 * mag]);
yticks(gca, [-mult * mag, 0, mult *  mag]);xlim(gca,[239,240])
xlim(gca,[239,240])

axes('Position',[pos1(1), pos2(2)+pos2(4)/3, pos2(3), pos2(4)/3]);
axN = plot(decimal_sol_s, BP_VBB_N./mag);
set(gca,'xticklabel',[])
ylabel('$V_{seis}$ [ms\textsuperscript{-1}]')

ymin = min(BP_VBB_N./mag);
ymax = max(BP_VBB_N./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])

axes('Position',[pos2(1), pos2(2), pos2(3), pos2(4)/3]);
axE = plot(decimal_sol_s, BP_VBB_E./mag);
set(gca,'xticklabel',[])

ymin = min(BP_VBB_E./mag);
ymax = max(BP_VBB_E./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])

set(axN, 'Color', colours(1,:));
set(axE, 'Color', colours(2,:));
set(axZ, 'Color', colours(3,:));

%---------------------------------------------
axes('Position',[pos3(1), pos3(2)+2*pos3(4)/3, pos3(3), pos3(4)/3]);
axZ = plot(decimal_sol_s, a_VBB_Z);
set(gca,'xticklabel',[])

ymin = min(a_VBB_Z);
ymax = max(a_VBB_Z);
mag = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag)),floor(ymax/mag));
mult2 = max(abs(floor(ymin/mag)),ceil(ymax/mag));
ylim([-mult2 * mag, mult2 * mag]);
yticks(gca, [-mult * mag, 0, mult *  mag]);xlim(gca,[239,240])
xlim(gca,[239,240])

axes('Position',[pos1(1), pos3(2)+pos3(4)/3, pos3(3), pos3(4)/3]);
axN = plot(decimal_sol_s, a_VBB_N./mag);
set(gca,'xticklabel',[])
ylabel('$a_{seis}$ [ms\textsuperscript{-2}]')

ymin = min(a_VBB_N./mag);
ymax = max(a_VBB_N./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])

axes('Position',[pos1(1), pos3(2), pos3(3), pos3(4)/3]);
axE = plot(decimal_sol_s, a_VBB_E./mag);

ymin = min(a_VBB_E./mag);
ymax = max(a_VBB_E./mag);
mag2 = 10^floor(log10(max(abs([ymin ymax]))));
mult = max(abs(ceil(ymin/mag2)),floor(ymax/mag2));
mult2 = max(abs(floor(ymin/mag2)),ceil(ymax/mag2));
mult2 = mult *1.5;
ylim([-mult2 * mag2, mult2 * mag2]);
yticks(gca, [-mult * mag2, 0, mult *  mag2]);
xlim(gca,[239,240])
xlabel('Decimal Sol [-]')

set(axZ, 'Color', colours(1,:));
set(axN, 'Color', colours(2,:));
set(axE, 'Color', colours(3,:));
%%
seis_windows = NaN(size(p_windows));
    for i = 1:length(pressure_peaks)
        [~,idx1] = min(abs(decimal_sol_s - decimal_sol(p_windows(i,1))));
        seis_windows(i,1) = idx1;
        [~,idx2] = min(abs(decimal_sol_s - decimal_sol(p_windows(i,2))));
        seis_windows(i,2) = idx2;
    end

ii = 1;

thetaObs = (sqrt((a_VBB_E(seis_windows(ii,1):seis_windows(ii,2))).^2 + (a_VBB_N(seis_windows(ii,1):seis_windows(ii,2))).^2))/3.71;
alphaObs = atan2(a_VBB_E(seis_windows(ii,1):seis_windows(ii,2)),(a_VBB_N(seis_windows(ii,1):seis_windows(ii,2))));
figure
plot(decimal_sol_s(seis_windows(ii,1):seis_windows(ii,2)),thetaObs)
hold on
plot([params_fit(2) params_fit(2)],[0,1.4e-8],'k--');
xlabel('Decimal Sol [-]')
ylabel('$\theta_{obs}$ [rad]')
%% MAGNETIC
ifg_20 = readtable('/media/david/Extreme SSD/ifg_data_calibrated/ifg_cal_SOL0239_20Hz_v06.tab','FileType','text');
ifg_2 = readtable('/media/david/Extreme SSD/ifg_data_calibrated/ifg_cal_SOL0239_20Hz_v06.tab','FileType','text');
ifg_p2 = readtable('/media/david/Extreme SSD/ifg_data_calibrated/ifg_cal_SOL0239_20Hz_v06.tab','FileType','text');

%%
file_sol = 239;

h = ifg_20.TLST;
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/24;
ifg_20.TLST = decimal_sol_ifg;

h = ifg_2.TLST;
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/24;
ifg_2.TLST = decimal_sol_ifg;

h = ifg_p2.TLST;
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/24;
ifg_p2.TLST = decimal_sol_ifg;

%%
figure
tiledlayout(3,1)
nexttile(1)
axZ = plot(ifg_2.TLST, ifg_2.B_down);
set(gca,'xticklabel',[])
xlim(gca,[239,240])
ylabel('$B_Z$ [nT]')

nexttile(2)
axN = plot(ifg_2.TLST, ifg_2.B_north);
set(gca,'xticklabel',[])
ylabel('$B_N$ [nT]')
xlim(gca,[239,240])

nexttile(3)
axE = plot(ifg_2.TLST, ifg_2.B_east);
xlim(gca,[239,240])
ylabel('$B_E$ [nT]')
xlabel('Decimal Sol [-]')

colours = lines(3);

set(axZ, 'Color', colours(1,:));
set(axN, 'Color', colours(2,:));
set(axE, 'Color', colours(3,:));
%%
mag_windows = NaN(size(p_windows));
    for i = 1:length(pressure_peaks)
        [~,idx1] = min(abs(ifg_2.TLST - decimal_sol(p_windows(i,1))));
        mag_windows(i,1) = idx1;
        [~,idx2] = min(abs(ifg_2.TLST - decimal_sol(p_windows(i,2))));
        mag_windows(i,2) = idx2;
    end
ii = 1;
colours = lines(3);

preevent_time = 200;
preevent_Z = ifg_2.B_down(mag_windows(ii,1)-preevent_time*2:mag_windows(ii,1));
pre_Z_med = median(preevent_Z);
preevent_N = ifg_2.B_north(mag_windows(ii,1)-preevent_time*2:mag_windows(ii,1));
pre_N_med = median(preevent_N);
preevent_E = ifg_2.B_east(mag_windows(ii,1)-preevent_time*2:mag_windows(ii,1));
pre_E_med = median(preevent_E);

figure
tiledlayout(3,1)
nexttile(1)
plot(ifg_2.TLST(mag_windows(ii,1):mag_windows(ii,2)),ifg_2.B_down(mag_windows(ii,1):mag_windows(ii,2))-pre_Z_med,'Color',colours(1,:))
hold on
plot([params_fit(2) params_fit(2)],[-1,1],'k--');
ylabel('$B_Z$ [nT]')
set(gca, 'XTickLabel', [])

nexttile(2)
plot(ifg_2.TLST(mag_windows(ii,1):mag_windows(ii,2)),ifg_2.B_north(mag_windows(ii,1):mag_windows(ii,2))-pre_N_med,'Color',colours(2,:))
hold on
plot([params_fit(2) params_fit(2)],[-1,1],'k--');

ylabel('$B_N$ [nT]')
set(gca, 'XTickLabel', [])

nexttile(3)
plot(ifg_2.TLST(mag_windows(ii,1):mag_windows(ii,2)),ifg_2.B_east(mag_windows(ii,1):mag_windows(ii,2))-pre_E_med,'Color',colours(3,:))
hold on
plot([params_fit(2) params_fit(2)],[-1,1],'k--');

ylabel('$B_E$ [nT]')
xlabel('Decimal Sol [-]')


%%

function [data_out, sample_rate] = lowpass_2Hz(data, sample_rate)
    
    if sample_rate == 2
        % If the sample rate is already 2 Hz, return the data as is
        data_out = data;
    else
        % Define the cutoff frequency
        cutoff_frequency = 2.0;  % Hz
        
        % Calculate the Nyquist frequency
        nyquist_frequency = sample_rate / 2.0;
        
        % Normalize the cutoff frequency
        normalized_cutoff = cutoff_frequency / nyquist_frequency;
        
        % Design the low-pass Butterworth filter (2nd order)
        [b, a] = butter(2, normalized_cutoff, 'low');
        
        % Apply the filter to the second column of the data (signal)
        filtered_data = filtfilt(b, a, data); % Apply zero-phase filtering
        
        % Create the output data by stacking time and filtered signal
        data_out =filtered_data;
    end
end

function [P] = lv_pressure(x,y,P0,x0,y0,R)
    r = sqrt((x - x0).^2 + (y - y0).^2);
    P = -abs(P0)./(1+((2 * r)/R).^2);
end