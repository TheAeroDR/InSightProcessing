%raw = readtable('dataout_1000s.csv');
%mag = readtable('magout_1000s.csv');
%pres = readtable('presout_1000s.csv');
%mag_bckgnd = readtable('backgroundout_1000s.csv');
%raw_PS = readtable('dataout_PS.csv');
%%
load("raw.mat")
%%
% decimal time in day of peaks
zulu_time = raw.p_lsr_peak_time-floor(raw.p_lsr_peak_time);
zulu_time_PS = raw_PS.p_lsr_peak_time-floor(raw_PS.p_lsr_peak_time);

%% PS compare
figure
histogram(hours(seconds(zulu_time*86400)),8:0.5:17,'Normalization','pdf')
hold on

histogram(hours(seconds(zulu_time_PS*86400)),8:0.5:17,'Normalization','pdf')

legend('Dataset','PS Dataset')
xlabel('LTST')
ylabel('Probability Density')

[h,p] = kstest2(hours(seconds(zulu_time*86400)), hours(seconds(zulu_time_PS*86400)));
display(h);display(p);

clear zulu_time_PS raw_PS
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
mag = mag(:,repelem(keep_check, 4));
mag_bckgnd = mag_bckgnd(:,repelem(keep_check, 4));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);

zulu_time = zulu_time(keep_check);

clear keep_check clear string_to_ignore column_names columns_to_keep
%%
figure
scatter3(hours(seconds(zulu_time*86400)),raw.p_lsr_delta_P,raw.p_lsr_fit,50,raw.p_lsr_fit,'filled')
cb = colorbar();
view(2)
xlim([0,24])
xticks([0,6,12,18,24])
xticklabels({'00:00', '06:00', '12:00', '18:00', '24:00'})
xlabel('LTST')
ylabel('$\Delta{}P$ [Pa]')

%% full set split
keep_check = zulu_time > 8/24 & zulu_time < 17/24 & raw.p_lsr_fit < 0.8 & raw.time_FWHM < prctile(raw.time_FWHM,99) & raw.time_FWHM > prctile(raw.time_FWHM,25);
%keep_check = true(size(zulu_time));

mag_real = mag(:,repelem(keep_check, 4));
mag_bckgnd_real = mag_bckgnd(:,repelem(keep_check, 4));
pres_real = pres(:,repelem(keep_check, 2));
raw_real = raw(keep_check, :);
zulu_time_real = zulu_time(keep_check);

clear keep_check

%%%
%spigadataset = readtable('spiga_dataset.txt');

%figure
%histogram(hours(seconds(zulu_time_real*86400)),6:0.5:18,'Normalization','pdf')
%hold on

%histogram(spigadataset.x_LTST_,8:0.5:17,'Normalization','pdf')


%legend('Dataset','Spiga Dataset')
%xlabel('LTST')
%ylabel('Probability Density')


%[h,p] = kstest2(spigadataset.x_LTST_, hours(seconds(zulu_time_real*86400)));
%display(h);display(p);


%%
tiledlayout(1,2)

nexttile
polarscatter(raw_real.azimuth, raw_real.miss_distance)
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';

nexttile
polarhistogram(raw_real.azimuth)
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';

%%
load("noise_times_LTST.mat")
sols = unique(raw_real.Sol);

for i = 1:length(raw_real.Sol)
    idx = find(floor(start_noise)==raw_real.Sol(i));
    if idx
        keep_check(i) = raw_real.peak_centre(i) < start_noise(idx) - (250/84600) || raw_real.peak_centre(i) > end_noise(idx) + (250/84600);
    else
        keep_check(i) = 0;5159
    end
end

mag_event = mag_real(:,repelem(keep_check, 4));
mag_bckgnd_event = mag_bckgnd_real(:,repelem(keep_check, 4));
pres_event = pres_real(:,repelem(keep_check, 2));
raw_event = raw_real(keep_check, :);

zulu_time = raw_event.peak_centre-floor(raw_event.peak_centre);
clear keep_check

% Loop over numeric columns and replace zeros with NaN
for i = 1:width(mag_event)
    mag_event{:,i}(mag_event{:,i} == 0) = NaN;
end

for i =1:length(raw_event.Sol)
    mag_event{:,(4*i)-3} = 87755.244 * (mag_event{:,(4*i)-3} - raw_event.peak_centre(i));
end

for i = 1:length(raw_event.peak_centre)
    mag_bckgnd_event{:,4*i -3} = 87755.244 * (mag_bckgnd_event{:,4*i-3} - nanmedian(mag_bckgnd_event{:,4*i-3}));
end


%some event have no magnetic data?
keep_check = ~isnan(raw_event.peak_Bz);

mag_event = mag_event(:,repelem(keep_check, 4));
mag_bckgnd_event = mag_bckgnd_event(:,repelem(keep_check, 4));
pres_event = pres_event(:,repelem(keep_check, 2));
raw_event = raw_event(keep_check, :);

zulu_time = raw_event.peak_centre-floor(raw_event.peak_centre);
clear keep_check

%%
colours = lines(2);
figure
plot(hours(seconds((start_noise - floor(start_noise))*86400)), floor(start_noise),'.',Color=colours(1,:))
hold on
plot(hours(seconds((end_noise - floor(end_noise))*86400)), floor(end_noise),'.',Color=colours(2,:))

temp = smooth(hours(seconds((start_noise - floor(start_noise))*86400)),0.5,'lowess');
temp2 = smooth(hours(seconds((end_noise - floor(end_noise))*86400)),0.5,'lowess');

plot(temp,floor(start_noise),Color=colours(1,:))
plot(temp2,floor(end_noise),Color=colours(2,:))

sol_temp = 1:800;
[noon, sunrise,sunset] = mars_daytimes_insight(sol_temp);

plot(noon, sol_temp)
%plot(sunrise, sol_temp)
%plot(sunset, sol_temp)
xlabel('LTST')
ylabel('Sol')

yyaxis right

yticks([7 57 114 174 240 306 371 438 505 572 639 706 773]/800)
yticklabels([300 330 0 30 60 90 120 150 180 210 240 270 300])
ylabel('$L_s$ [deg]')
%set(gca,YColor,[1 0 0]);
%%
figure
hold on
for i =1:length(raw_event.Sol)
    plot(mag_event{:,(4*i)-3}, mag_event{:,4*i})
end
%% Bz
%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';
pol_win = 10;
for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_event{:,(4*i)-3}) & ~isnan(mag_event{:,4*i});
    xValid = mag_event{validIndices,(4*i)-3};
    yValid = mag_event{validIndices,4*i};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_event(:,4*i) = (mag_event(:,4*i))-fitted_y;

    yValid = mag_event{validIndices,4*i};
    event_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end

%%
figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(mag_event{:,4*i -3},mag_event{:,4*i})
end

pol_idx = abs(interp_range)<pol_win;
for i = 1:length(raw_event.peak_centre)
    polScore = median(event_align(pol_idx,i), 'omitnan');
    pol(i) = sign(polScore);
end

figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(interp_range,pol(i) * event_align(:,i))
end

mag_med = median((pol(i) * event_align),2);
figure
plot(interp_range,mag_med,'k',LineWidth=2)
xlabel('Time [s]')
ylabel('$B_Z$ [nT]')

%%
%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';

for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_bckgnd_event{:,(4*i)-3}) & ~isnan(mag_bckgnd_event{:,4*i});
    xValid = mag_bckgnd_event{validIndices,(4*i)-3};
    yValid = mag_bckgnd_event{validIndices,4*i};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_bckgnd_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_bckgnd_event(:,4*i) = (mag_bckgnd_event(:,4*i))-fitted_y;

    yValid = mag_bckgnd_event{validIndices,4*i};
    bckgnd_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
mag_bckgnd_med = (1.58 * iqr(pol.*bckgnd_align,2)) ./ sqrt(height(bckgnd_align));

patch([interp_range' fliplr(interp_range')],[abs(mag_bckgnd_med)' -fliplr(abs(mag_bckgnd_med)')],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7]);


back_x = mag_bckgnd_event{:,4:4:end};
histogram(back_x)
kstest(back_x)

%% Bn
%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';
pol_win = 10;
for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_event{:,(4*i)-3}) & ~isnan(mag_event{:,4*i-2});
    xValid = mag_event{validIndices,(4*i)-3};
    yValid = mag_event{validIndices,4*i-2};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_event(:,4*i-2) = (mag_event(:,4*i-2))-fitted_y;

    yValid = mag_event{validIndices,4*i-2};
    event_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end

figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(mag_event{:,4*i -3},mag_event{:,4*i-2})
end

%pol_idx = abs(interp_range)<pol_win;
%for i = 1:length(raw_event.peak_centre)
%    polScore = median(event_align(pol_idx,i), 'omitnan');
%    pol(i) = sign(polScore);
%end

figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(interp_range,pol(i) * event_align(:,i))
end

mag_med = median((pol(i) * event_align),2);
figure
plot(interp_range,mag_med,'k',LineWidth=2)
xlabel('Time [s]')
ylabel('$B_N$ [nT]')

%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';

for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_bckgnd_event{:,(4*i)-3}) & ~isnan(mag_bckgnd_event{:,4*i-2});
    xValid = mag_bckgnd_event{validIndices,(4*i)-3};
    yValid = mag_bckgnd_event{validIndices,4*i-2};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_bckgnd_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_bckgnd_event(:,4*i-2) = (mag_bckgnd_event(:,4*i-2))-fitted_y;

    yValid = mag_bckgnd_event{validIndices,4*i-2};
    bckgnd_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
mag_bckgnd_med = (1.58 * iqr(pol.*bckgnd_align,2)) ./ sqrt(height(bckgnd_align));

patch([interp_range' fliplr(interp_range')],[abs(mag_bckgnd_med)' -fliplr(abs(mag_bckgnd_med)')],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7]);


back_x = mag_bckgnd_event{:,2:4:end};
back_x = back_x(:);
back_x = (back_x-nanmean(back_x))/nanstd(back_x);
histogram(back_x)
[h,p] = kstest(back_x)

%% Be
%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';
pol_win = 10;
for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_event{:,(4*i)-3}) & ~isnan(mag_event{:,4*i-1});
    xValid = mag_event{validIndices,(4*i)-3};
    yValid = mag_event{validIndices,4*i-1};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_event(:,4*i-1) = (mag_event(:,4*i-1))-fitted_y;

    yValid = mag_event{validIndices,4*i-1};
    event_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end

figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(mag_event{:,4*i -3},mag_event{:,4*i-1})
end

%pol_idx = abs(interp_range)<pol_win;
%for i = 1:length(raw_event.peak_centre)
%    polScore = median(event_align(pol_idx,i), 'omitnan');
%    pol(i) = sign(polScore);
%end

figure
hold on
for i = 1:length(raw_event.peak_centre)
    plot(interp_range,pol(i) * event_align(:,i))
end

mag_med = median((pol(i) * event_align),2);
figure
plot(interp_range,mag_med,'k',LineWidth=2)
xlabel('Time [s]')
ylabel('$B_E$ [nT]')

%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';

for i = 1:length(raw_event.peak_centre)
    %linear detrend
    validIndices = ~isnan(mag_bckgnd_event{:,(4*i)-3}) & ~isnan(mag_bckgnd_event{:,4*i-1});
    xValid = mag_bckgnd_event{validIndices,(4*i)-3};
    yValid = mag_bckgnd_event{validIndices,4*i-1};

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_bckgnd_event{:,4*i-3};
    fitted_y = feval(fitresult, fitted_x);

    mag_bckgnd_event(:,4*i-1) = (mag_bckgnd_event(:,4*i-1))-fitted_y;

    yValid = mag_bckgnd_event{validIndices,4*i-2};
    bckgnd_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
mag_bckgnd_med = (1.58 * iqr(pol.*bckgnd_align,2)) ./ sqrt(height(bckgnd_align));

patch([interp_range' fliplr(interp_range')],[abs(mag_bckgnd_med)' -fliplr(abs(mag_bckgnd_med)')],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7]);

back_x = mag_bckgnd_event{:,3:4:end};
histogram(back_x)
kstest(back_x)

%%
a = max(event_align.*pol);

figure
tiledlayout(2,2)
nexttile(1)
plot(raw_event.p_lsr_delta_P,a,'.')
xlabel('$P_{obs}$ [Pa]')
ylabel('$B_Z$ [nT]')
hold on

[x_sorted, sortIdx] = sort(raw_event.p_lsr_delta_P);
y_sorted = a(sortIdx);
smoothed_y = smooth(x_sorted, y_sorted, 0.01, 'loess');
plot(x_sorted, smoothed_y)


nexttile(2)
plot(raw_event.miss_distance,a,'.')
xlabel('$b$ [m]')
hold on

[x_sorted, sortIdx] = sort(raw_event.miss_distance);
y_sorted = a(sortIdx);
smoothed_y = smooth(x_sorted, y_sorted, 0.01, 'loess');
plot(x_sorted, smoothed_y)

nexttile(3)
plot(raw_event.time_FWHM,a,'.')
xlabel('$\tau_{obs}$ [Pa]')
ylabel('$B_Z$ [nT]')
hold on

[x_sorted, sortIdx] = sort(raw_event.time_FWHM);
y_sorted = a(sortIdx);
smoothed_y = smooth(x_sorted, y_sorted, 0.01, 'loess');
plot(x_sorted, smoothed_y)

nexttile(4)
plot(raw_event.backg_V,a,'.')
xlabel('$V_{BGR}$ [ms\textsuperscript{-1}]')
hold on

[x_sorted, sortIdx] = sort(raw_event.backg_V);
y_sorted = a(sortIdx);
smoothed_y = smooth(x_sorted, y_sorted, 0.01, 'loess');
plot(x_sorted, smoothed_y)

%%
%nexttile(2)
bins = [8,9,10,11,12,13,14,15,16,17,18]*3600/86400;
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

iosr.statistics.boxPlot(midpointtimes,boxplotdata','scaleWidth',true,'notch',true,'showOutliers',false)
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

%% functions
function [n,sr,ss] = mars_daytimes_insight(sol_list)
    % Parameters
    Ls0 = 295.5;            % InSight landing solar longitude [deg]
    sols_per_year = 668.6;  % mean tropical year [sols]
    
    % Site latitude
    phi = deg2rad(4.5);     % InSight latitude
    eps = deg2rad(25.19);   % Mars obliquity
    sol_length = 24.6597;   % mean sol length [hours]
    i = 1;
    for sol = sol_list
        % Compute Ls for this sol (wrap 0–360)
        Ls = Ls0 + (360.0/sols_per_year)*sol;
        Ls = mod(Ls,360.0);
        
        % Solar declination
        delta = asin(sin(eps) * sin(deg2rad(Ls)));
        
        % Hour angle at sunrise/sunset
        cosH = -tan(phi) * tan(delta);
        cosH = max(min(cosH,1),-1);  % clip to [-1,1]
        H = acos(cosH);
        
        % Convert to hours of LTST
        daylength_hours = (2*H) * (sol_length/(2*pi));
        half_day = daylength_hours/2;
        
        ltst_noon = 12.0;
        ltst_sunrise = ltst_noon - half_day;
        ltst_sunset  = ltst_noon + half_day;
        
        n(i) = ltst_noon;
        sr(i) = ltst_sunrise;
        ss(i) = ltst_sunset;
        i = i + 1;
    end
end