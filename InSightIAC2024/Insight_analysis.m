addpath("IAC_data/)

raw = readtable('dataout.csv');
mag = readtable('magout.csv');
pres = readtable('presout.csv');
mag_bckgnd = readtable('backgroundout.csv');
%% do not use this one
raw = readtable('dataout_mov_av.csv');
mag = readtable('magout_mov_av.csv');
pres = readtable('presout_mov_av.csv');
%%
addpath("IAC_data/)

raw = readtable('dataout_500s.csv');
mag = readtable('magout_500s.csv');
pres = readtable('presout_500s.csv');
mag_bckgnd = readtable('backgroundout_500s.csv');
%%
addpath("IAC_data/)

raw = readtable('dataout_1000s.csv');
mag = readtable('magout_1000s.csv');
pres = readtable('presout_1000s.csv');
mag_bckgnd = readtable('backgroundout_1000s.csv');
%% filtering
string_to_ignore = 'Unnamed_';
column_names = mag.Properties.VariableNames;
columns_to_keep = ~startsWith(column_names, string_to_ignore);
mag = mag(:, columns_to_keep);

column_names = pres.Properties.VariableNames;
columns_to_keep = ~startsWith(column_names, string_to_ignore);
pres = pres(:, columns_to_keep);

column_names = mag_bckgnd.Properties.VariableNames;
columns_to_keep = ~startsWith(column_names, string_to_ignore);
mag_bckgnd = mag_bckgnd(:, columns_to_keep);

% remove NaN dias
keep_check = ~isnan(raw.D);
mag = mag(:,repelem(keep_check, 2));
mag_bckgnd = mag_bckgnd(:,repelem(keep_check, 2));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);

% decimal time in day of peaks
zulu_time = raw.peak_centre-floor(raw.peak_centre);

clear keep_check clear string_to_ignore column_names columns_to_keep
%%
%keep 'best' events
uq_dP = quantile(raw.delta_P,0.25);
uuq_dP = quantile(raw.delta_P, 0.05);
lq_D = quantile(raw.D,0.25);
llq_D = quantile(raw.D, 0.05);
keep_check = raw.delta_P<uq_dP & raw.delta_P > uuq_dP & raw.D<lq_D & raw.D > llq_D;

mag_sort = mag(:,repelem(keep_check, 2));
pres_sort = pres(:,repelem(keep_check, 2));
raw_sort = raw(keep_check, :);


clear uq_dP lq_D uuq_dP llq_D
clear keep_check

% %% remove spurious 0s from filtered data
dataArray = table2array(mag_sort);
zeroIndices = (dataArray == 0);
dataArray(zeroIndices) = NaN;
mag_sort = array2table(dataArray, 'VariableNames', mag_sort.Properties.VariableNames);

dataArray = table2array(pres_sort);
zeroIndices = (dataArray == 0);
dataArray(zeroIndices) = NaN;
pres_sort = array2table(dataArray, 'VariableNames', pres_sort.Properties.VariableNames);

clear dataArray zeroIndices

%% full set split
keep_check = zulu_time > 0.333333333333 & zulu_time < 0.75  & raw.delta_P>-10;% & raw.D < 500;

mag_real = mag(:,repelem(keep_check, 2));
mag_bckgnd_real = mag_bckgnd(:,repelem(keep_check, 2));
pres_real = pres(:,repelem(keep_check, 2));
raw_real = raw(keep_check, :);
zulu_time_real = zulu_time(keep_check);

clear keep_check
%% linear detrend and cosine taper filtered data
for i = 1:height(raw_sort)
    x = table2array(mag_sort(:,2*i - 1)) - raw_sort.peak_centre(i);
    y = abs(table2array(mag_sort(:, 2*i)));

    validIndices = ~isnan(x) & ~isnan(y);
    xValid = x(validIndices);
    yValid = y(validIndices);
    coefficients = polyfit(xValid, yValid, 1);
    yDetrend = yValid - (coefficients(1) * xValid + coefficients(2));
    
    N = length(yDetrend);
    t = linspace(0, 1, N)';
    taper = (1 - cos(2*pi*t)) / 2;
    yCosTap = yDetrend .* taper;

    mag_sort{validIndices, 2*i} = yDetrend;

end

clear validIndices coefficients x y xValid yValid yDetrend N alpha t taper yCosTap

% %% linear detrend and cosine taper background data
for i = 1:height(raw_sort)
    x = table2array(mag_bckgnd(:,2*i - 1)) - raw_sort.peak_centre(i);
    y = abs(table2array(mag_bckgnd(:, 2*i)));

    validIndices = ~isnan(x) & ~isnan(y);
    xValid = x(validIndices);
    yValid = y(validIndices);
    coefficients = polyfit(xValid, yValid, 1);
    yDetrend = yValid - (coefficients(1) * xValid + coefficients(2));
    
    N = length(yDetrend);
    t = linspace(0, 1, N)';
    taper = (1 - cos(2*pi*t)) / 2;
    yCosTap = yDetrend .* taper;

    mag_bckgnd{validIndices, 2*i} = yDetrend;

end

clear validIndices coefficients x y xValid yValid yDetrend N alpha t taper yCosTap

disp("detrended");

%%
figure(1)
tiledlayout(2,1)
nexttile
hold on
for i = 1:height(raw_sort)
plot(table2array(mag_sort(:,2*i - 1)) - raw_sort.peak_centre(i),table2array(mag_sort(:, 2 * i)))
end
nexttile
hold on
for i = 1:height(raw_sort)
plot(table2array(pres_sort(:,2*i - 1)) - raw_sort.peak_centre(i),table2array(pres_sort(:, 2 * i)))
end
%%
clear mag_align

mag_align = table2array(mag_sort);
mag_align = mag_align(:,2:2:end);

min_rate = min(unique(raw_sort.mag_rate));
av_window = raw_sort.mag_rate./min_rate;
averaged_data = [];
for i = 1:height(raw_sort)
    data = table2array(mag_sort(:,2 * i));
    if av_window(i) ~= 1
        low = 1;
        averaged_column = [];
        while low < length(data)
            high = min(low + av_window(i), length(data));
            averaged_value = nanmean(data(low:high-1));
            if isnan(averaged_value)
                break; % Exit the loop if mean is NaN
            end
            averaged_column = [averaged_column; averaged_value];
            low = high;
        end
        averaged_data{i} = averaged_column;
    else
        averaged_data{i} = data(~isnan(data));
    end
end
% Define a function to pad each cell
pad_func = @(x) padarray(x, [50 - length(x), 0], NaN, 'post');

for i = 1:length(averaged_data)
    index = min(find(abs(averaged_data{i}) < 1e-5),length(averaged_data{i}));
    if isempty(index)
        index = length(averaged_data{i});
    elseif length(index)>1 && index(2)-index(1) == 1 
        index = index(1);
    end
    averaged_data{i} = averaged_data{i}(1:index);
end
f1 = @(x) length(x)>50;
averaged_data = averaged_data(~cellfun(f1,averaged_data));


% Apply the padding function to each cell in averaged_data
averaged_data = cellfun(pad_func, averaged_data, 'UniformOutput', false);
averaged_data = cell2mat(averaged_data);
mag_med = nanmean(averaged_data,2);
mag_med = nanmedian(averaged_data,2);
%mag_med = exp(sum(log(abs(averaged_data)), 2,"omitnan") ./ length(averaged_data)) .* (prod(averaged_data,2,"omitnan")./abs(prod(averaged_data,2,"omitnan")));
%temp = (1/length(averaged_data)) * sum((nthroot(abs(averaged_data),4)) .* averaged_data./abs(averaged_data),2,"omitnan");
%mag_med = (abs(temp) .^4) .* (temp./abs(temp));

mag_med(end-1:end) = NaN;


mag_times = table2array(mag_sort(:,1)) - raw_sort.peak_centre(1);
mag_times = mag_times(~isnan(mag_times));
mag_times = interp1(1:48,mag_times,1:50,'linear','extrap')';

figure(1)
tiledlayout(2,1)
nexttile(2)
plot(mag_times, mag_med,'k','LineWidth',1.5)

clear min_rate av_window low data averaged_column averaged_data averaged_value high mag_align pad_func temp

% %%
clear pres_align

pres_align = table2array(pres_sort);
pres_align = pres_align(:,2:2:end);

min_rate = min(unique(raw_sort.pres_rate));
av_window = raw_sort.pres_rate./min_rate;
averaged_data = [];
for i = 1:height(raw_sort)
    data = table2array(pres_sort(:,2 * i));
    if av_window(i) ~= 1
        low = 1;
        averaged_column = [];
        while low < length(data)
            high = min(low + av_window(i), length(data));
            averaged_value = nanmean(data(low:high-1));
            if isnan(averaged_value)
                break; % Exit the loop if mean is NaN
            end
            averaged_column = [averaged_column; averaged_value];
            low = high;
        end
        averaged_data{i} = averaged_column;
    else
        averaged_data{i} = data(~isnan(data));
    end
end
averaged_data = cell2mat(averaged_data);
pres_med = nanmean(averaged_data,2);

pres_times = table2array(pres_sort(:,1)) - raw_sort.peak_centre(1);
pres_times = pres_times(~isnan(pres_times));

nexttile(1)
plot(pres_times, pres_med,'k','LineWidth',1.5)

clear min_rate av_window low data averaged_column averaged_data averaged_value high pres_align

%%
clear mag_align_bckgnd

mag_times = table2array(mag_sort(:,1)) - raw_sort.peak_centre(1);
mag_times = mag_times(~isnan(mag_times));
mag_times = interp1(1:48,mag_times,1:50,'linear','extrap')';

mag_align_bckgnd = table2array(mag_bckgnd);
mag_align_bckgnd = mag_align_bckgnd(:,2:2:end);

min_rate = min(unique(raw_sort.mag_rate));
av_window = raw_sort.mag_rate./min_rate;
averaged_data = [];
for i = 1:height(raw_sort)
    data = table2array(mag_bckgnd(:,2 * i));
    if av_window(i) ~= 1
        low = 1;
        averaged_column = [];
        while low < length(data)
            high = min(low + av_window(i), length(data));
            averaged_value = nanmean(data(low:high-1));
            if isnan(averaged_value) 
                break; % Exit the loop if mean is NaN
            end
            averaged_column = [averaged_column; averaged_value];
            low = high;
        end
        averaged_data{i} = averaged_column;
    else
        averaged_data{i} = data(~isnan(data));
    end
end

mag_align_bckgnd = cell2mat(averaged_data);

mag_bckgnd_med = (1.58 * iqr(abs(mag_align_bckgnd),2)) ./ sqrt(height(mag_align_bckgnd));

nexttile(2)
patch([mag_times' fliplr(mag_times')],[abs(mag_bckgnd_med(1:50))' -fliplr(abs(mag_bckgnd_med(1:50))')],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7]);
%%
figure(1)
%tiledlayout(2,1)
%nexttile(1)
plot(raw_real.D, -raw_real.delta_P,'.')

ylabel("Delta-P $[Pa]$")
xlabel("Diameter $[m]$")
set(gca,'XAxisLocation','top')

%D_P_fit = smooth(raw_real.D, raw_real.delta_P, 0.4, 'lowess');
%[sorted_D, sort_idx] = sort(raw_real.D);
%D_P_fit = D_P_fit(sort_idx);
hold on
%plot(sorted_D, D_P_fit,'LineWidth',1.5)


[xData, yData] = prepareCurveData(raw_real.D, raw_real.delta_P);

% Set up fittype and options.
ft = fittype( 'power1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.655098003973841 -2];

% Fit model to data.
fitresult = fit( xData, yData, ft);

fitted_D = linspace(min(raw_real.D), max(raw_real.D), 100);
fitted_delta_P = feval(fitresult, fitted_D);

plot(fitted_D, fitted_delta_P);
%%
figure(2)
plot(zulu_time_real,-1*raw_real.delta_P,'.')
hold on
ylabel("Delta-P $[Pa]$")
xlabel("Decimal Day [-]")

bins = [6,7,8,9,10,11,12,13,14,15,16,17,18]*3600/88775.2440;
bindex = discretize(zulu_time_real,bins);

for i = 1:length(bins)-1
    lobfff(i) = median(-1*raw.delta_P(bindex==i));
end
lobfx=[23400:3600:63000]/88775.255;

plot(lobfx,lobfff)

ToD_P_fit = smooth(zulu_time_real, -1*raw_real.delta_P, 0.5, 'lowess');
[sorted_zulu, sort_idx] = sort(zulu_time_real);
ToD_P_fit = ToD_P_fit(sort_idx);
plot(sorted_zulu, ToD_P_fit, 'LineWidth',1.5)
%%
%nexttile(2)
bins = [8,9,10,11,12,13,14,15,16,17,18]*3600/88775.2440;
bindex = discretize(zulu_time_real,bins);
boxplotdata=[];
len=0;
midpointtimes = [8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5];
for i = 1:length(bins)-1
    temp = (-1*raw_real.delta_P(bindex==i));
    len = max(len,length(temp));
end
for i =1:length(bins)-1
    temp = (-1*raw_real.delta_P(bindex==i));
    boxplotdata(i,:) = padarray(temp,len-length(temp),NaN,'post');
end

iosr.statistics.boxPlot(midpointtimes,boxplotdata','scaleWidth',true,'notch',true)
ylabel("$\Delta$P [Pa]")
xlabel("LTST [hr]")

Jackson = [9, 0.79414
10, 1.03889
11, 1.18501
12, 1.30412
13, 1.26315
14, 1.16943
15, 0.85712
16, 0.72725];

hold on
plot(Jackson(:,1),Jackson(:,2))
%%
figure(4)
plot(zulu_time_real,raw_real.backg_V,'.')

ylabel("Velocity $[m/s]$")
xlabel("Time of Day $[-]$")

ToD_V_fit = smooth(zulu_time_real,raw_real.backg_V, 0.2, 'lowess');
hold on
[sorted_zulu, sort_idx] = sort(zulu_time_real);
ToD_V_fit = ToD_V_fit(sort_idx);
plot(sorted_zulu, ToD_V_fit, 'LineWidth',1.5)

[xData, yData] = prepareCurveData(zulu_time_real, raw_real.backg_V);

% Set up fittype and options.
ft = fittype( 'poly2' );

% Fit model to data.
fitresult = fit( xData, yData, ft);


plot(fitresult);
%%
figure(5)
plot(raw.time_FWHM.*raw.backg_V,raw.peak_Bz,'.')

ylabel("Magnetic Field $[nT]$")
xlabel("FWHM Diameter $[m]$")

M_D_fit = smooth(raw.time_FWHM.*raw.backg_V,raw.peak_Bz, 0.1, 'lowess');
[sorted_FWHM_D, sort_idx] = sort(raw.time_FWHM.*raw.backg_V);
M_D_fit = M_D_fit(sort_idx);
hold on
plot(sorted_FWHM_D, M_D_fit,'LineWidth',1.5)

%%
clear mag_align
interp_range = linspace(-1.6,1.6,3840)';
for i = 1:height(raw)
    mag_event= mag{:,[(2*i)-1, 2*i]};
    mag_event = mag_event(~isnan(mag_event(:,1)),:);
    mag_event = mag_event(~mag_event(:,1)==0,:);
    if mag_event
        if length(mag_event(:,1)) == length(unique(mag_event(:,1)))
            mag_align{i} = interp1(24*60*(mag_event(:,1)-raw.peak_centre(i)), mag_event(:,2),interp_range);
        else
            display(i)
        end
    end
end

idxPurge = cellfun(@isempty, mag_align);

mag_align = mag_align(~(idxPurge));

mag_align = cell2mat(mag_align);

figure
hold on
for i = 1:length(mag_align(1,:))
    plot(interp_range,(mag_align(:,i)))
end
mag_med = abs(prod(mag_align,2)).^(1/2850);
mag_med = median(abs(mag_align),2);
plot(interp_range,mag_med,'k',LineWidth=2)

%%
clear mag_P1
figure
hold on
mag_align = table2array(mag);
mag_align = mag_align(:,2:2:end);
for i = 1:length(mag_align(1,:))
    Fs = raw.mag_rate(i); %Hz
    n = 20000;
    L = length(mag_align(~isnan(mag_align(:,i)),i));
    f = 0:(Fs/n):(Fs/2-Fs/n);
    mag_fft = fft(mag_align(~isnan(mag_align(:,i)),i),n);
    P2 = abs(mag_fft/L);
    P1 = P2(1:n/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    mag_P1{i} = P1;
end
mag_P1 = cell2mat(mag_P1);

for i = 1:length(mag_align(1,:))
    plot(f,mag_P1(1:1000,i))
end

xlabel("Frequency [Hz]")
ylabel("P1(f)")

%%
figure
hold on
mag_align = table2array(mag);
mag_align = mag_align(:,2:2:end);
for i = 1:length(mag_align(1,:))
    Fs = raw.mag_rate(i);
    pspectrum(mag_align(~isnan(mag_align(:,i)),i),Fs);
end
%%
% Define the data points
x = raw.D;
y = raw.miss_distance;
z = raw.delta_P;

% Plot the surface without mesh
figure;
scatter3(x, y, z, 'filled'); % Plot original data points
hold on;

xlabel('D');
ylabel('Miss Distance');
zlabel('Delta P');

%%
% Define the data points
x = raw.D;
y = raw.miss_distance;
z = raw.delta_P;

[xData, yData, zData] = prepareSurfaceData( x, y, z );

% Set up fittype and options.
ft = fittype( 'lowess' );

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft, 'Normalize', 'on' );

% Plot fit with data.
figure

scatter3(x,y,z,20,raw.p_lor_fit,'filled')
colormap('jet');

hold on
h = plot( fitresult);
h.FaceColor= "none";
h.EdgeColor = "k";

caxis([min(raw.p_lor_fit), max(raw.p_lor_fit)]);
cbar = colorbar;
cbar.Limits = caxis;
cbar.Location = "northoutside";

% Label axes
xlabel('Diameter $[m]$');
ylabel( 'Closest Approach $[m]$');
zlabel( '$\Delta$ P $[Pa]$');

%%
tiledlayout(1,2)
nexttile
polarscatter(raw_real.azimuth,raw_real.miss_distance)
nexttile
polarhistogram(raw.azimuth)

%%

set(gca,'XTick',...
    [0.33333333 0.375 0.416666666 0.4583333 0.5 0.54166666667 0.58333333 0.625 0.66666667 0.708333333 0.75],...
    'XTickLabel',{'8','9','10','11','12','13','14','15','16','17','18'});

%%
spiga = [0.0830401328642126, 0.629761007617612, 1.55832249331599, 3.07104491367186, 4.22280675649081, 5.14200822721316,...
    5.33760854017366, 5.56560890497425, 5.30040848065357, 4.69296750874801, 4.02504644007431, 3.01512482419972, 2.00040320064512,...
    1.67016267226028, 0.965041544066471	0.562320899713440	0.215760345216552	0.00144000230400369];

% [Bar0, 0.00346; Bar1, 0.02624; Bar2, 0.06493; Bar3, 0.12796; Bar4, 0.17595; 
% Bar5, 0.21425; Bar6, 0.22240; Bar7, 0.23190; Bar8, 0.22085; Bar9, 0.19554; 
% Bar10, 0.16771; Bar11, 0.12563; Bar12, 0.08335; Bar13, 0.06959; Bar14, 0.04021; 
% Bar15, 0.02343; Bar16, 0.00899; Bar17, 0.00006]

mine = (1/(length(zulu_time_real)*(1/48))) * [15,36,104,199,234,288,270,256,234,184,179,152,123,102,95,33,26,11,2,1];

[h,p] = kstest2(spiga, mine);
display(h);display(p);
figure
histogram(zulu_time_real,'Normalization','pdf')
hold on


spiga_x = 1/24 * [8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75];
plot(spiga_x,spiga)
%%
load("noise_times.mat")
sols = unique(raw_real.Sol);

for i = 1:length(raw_real.Sol)
    idx = find(floor(start_noise)==raw_real.Sol(i));
    if idx
        keep_check(i) = raw_real.peak_centre(i) < start_noise(idx) - (250/84600) || raw_real.peak_centre(i) > end_noise(idx) + (250/84600);
    else
        keep_check(i) = 0;
    end
end

mag_event = mag_real(:,repelem(keep_check, 2));
mag_bckgnd_event = mag_bckgnd_real(:,repelem(keep_check, 2));
pres_event = pres_real(:,repelem(keep_check, 2));
raw_event = raw_real(keep_check, :);

zulu_time = raw_event.peak_centre-floor(raw_event.peak_centre);
clear keep_check
%% 100s data set filtering
keep_check = true(1,length(raw_event.peak_centre));
keep_check([1251,1252,1256,1257,1258])=0;

mag_event = mag_event(:,repelem(keep_check, 2));
mag_bckgnd_event = mag_bckgnd_event(:,repelem(keep_check, 2));
pres_event = pres_event(:,repelem(keep_check, 2));
raw_event = raw_event(keep_check, :);
interp_range = linspace(-100,100,401)';
%% 100s data set filtering NAN EVENTS
keep_check = true(1,length(raw_event.peak_centre));
keep_check([828,840,841,842,843,844,845,846,847,848])=0;

mag_event = mag_event(:,repelem(keep_check, 2));
mag_bckgnd_event = mag_bckgnd_event(:,repelem(keep_check, 2));
pres_event = pres_event(:,repelem(keep_check, 2));
raw_event = raw_event(keep_check, :);
interp_range = linspace(-100,100,401)';
%% 250s data set filtering
keep_check = true(1,length(raw_event.peak_centre));
keep_check([9,10,12,11,13,14,15,18,19,20,21,793,989,1112,1120,1121,1122,1128,1129,1130,1131,1132 ...
    1221,1222,1223,1224,1225,1427,1428,1429,1430,1431,1432,1433,1434,1435,1492,1493,1494, ...
    1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1692,1693,1694,1710,1711,1712])=0;
mag_event = mag_event(:,repelem(keep_check, 2));
mag_bckgnd_event = mag_bckgnd_event(:,repelem(keep_check, 2));
pres_event = pres_event(:,repelem(keep_check, 2));
raw_event = raw_event(keep_check, :);
interp_range = linspace(-250,250,1001)';
%% 250s data set filtering NAN EVENTS
keep_check = true(1,length(raw_event.peak_centre));
keep_check([840,841,842,843,844,845])=0;
mag_event = mag_event(:,repelem(keep_check, 2));
mag_bckgnd_event = mag_bckgnd_event(:,repelem(keep_check, 2));
pres_event = pres_event(:,repelem(keep_check, 2));
raw_event = raw_event(keep_check, :);
interp_range = linspace(-250,250,1001)';
%%
clear align bckgnd palign
events = table2array(mag_event);
pevents = table2array(pres_event);
backgrounds = table2array(mag_bckgnd_event);
%filter out 0s
events(events==0)=NaN;
backgrounds(backgrounds==0)=NaN;
pevents(pevents==0) = NaN;

%remove peak centre times
for i = 1:length(raw_event.peak_centre)
    events(:,1+2*(i-1)) = 88775.2440 .* (events(:,1+2*(i-1))- raw_event.peak_centre(i));
    pevents(:,1+2*(i-1)) = 88775.2440 .* (pevents(:,1+2*(i-1))- raw_event.peak_centre(i));
    backgrounds(:,1+2*(i-1))=88775.2440 .* (backgrounds(:,1+2*(i-1)) - 0.5*(max(backgrounds(:,1+2*(i-1)))+min(backgrounds(:,1+2*(i-1)))));
end

%linear detrend and resample to same times
%interp_range = linspace(-500,500,2001);
for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(events(:,1+2*(i-1))) & ~isnan(events(:,2*i));
    xValid = events(validIndices,1+2*(i-1));
    yValid = abs(events(validIndices,2*i));
    %if i == 1161 || i ==1162
    %    [xValid, idx] = unique(xValid);
    %    yValid = yValid(idx);
    %end
    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = events(:,1+2*(i-1));
    fitted_y = feval(fitresult, fitted_x);

    events(:,2*i) = abs(events(:,2*i))-fitted_y;
    
    yValid = (events(validIndices,2*i));
    %if i == 1161 || i== 1162
    %    yValid = yValid(idx);
    %end
    align(:,i) = interp1(xValid,yValid,interp_range,'linear','extrap');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end

for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(backgrounds(:,1+2*(i-1))) & ~isnan(backgrounds(:,2*i));
    xValid = backgrounds(validIndices,1+2*(i-1));
    yValid = abs(backgrounds(validIndices,2*i));
    %if i == 1161 || i ==1162
    %    [xValid, idx] = unique(xValid);
    %    yValid = yValid(idx);
    %end
    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = backgrounds(:,1+2*(i-1));
    fitted_y = feval(fitresult, fitted_x);

    backgrounds(:,2*i) = abs(backgrounds(:,2*i))-fitted_y;
    
    yValid = (backgrounds(validIndices,2*i));
    %if i == 1161 || i== 1162
    %    yValid = yValid(idx);
    %end
    bckgnd(:,i) = interp1(xValid,yValid,interp_range,'linear','extrap');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end

for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(pevents(:,1+2*(i-1))) & ~isnan(pevents(:,2*i));
    xValid = pevents(validIndices,1+2*(i-1));
    yValid = (pevents(validIndices,2*i));

    palign(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
%%
figure
hold on
%for i = 1:length(raw_event.peak_centre)
%    plot(interp_range,align(:,i))
%end
summary = (1.58 * iqr((bckgnd),2)) ./ sqrt(length(bckgnd));
mag_med = median((align),2);
plot(interp_range,mag_med)
%smooth_fit = smooth(interp_range,mag_med, 0.03, 'lowess');
hold on 
%plot(interp_range,smooth_fit,'LineWidth',1.5)
patch([interp_range' flip(interp_range)'],[summary' flip(-summary)'],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7])
%%
figure
hold on
selected = find(raw_event.peak_Bz>0.35);
selected = find(raw_event.backg_V>quantile(raw_event.backg_V,0.80));
for i = selected
    %plot(interp_range,align(:,i))
    summed = align(:,i);
end
mag_med = median((summed),2);
plot(interp_range,mag_med)
smooth_fit = smooth(interp_range,mag_med, 0.1, 'lowess');
plot(interp_range,smooth_fit,'LineWidth',1.5)