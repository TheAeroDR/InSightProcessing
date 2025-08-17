raw = readtable('dataout.csv');
mag = readtable('magout.csv');
pres = readtable('presout.csv');

% filtering
string_to_ignore = 'Unnamed_';
column_names = mag.Properties.VariableNames;
columns_to_keep = ~startsWith(column_names, string_to_ignore);
mag = mag(:, columns_to_keep);

column_names = pres.Properties.VariableNames;
columns_to_keep = ~startsWith(column_names, string_to_ignore);
pres = pres(:, columns_to_keep);

%column_names = mag_bckgnd.Properties.VariableNames;
%columns_to_keep = ~startsWith(column_names, string_to_ignore);
% mag_bckgnd(:, columns_to_keep);

% remove NaN dias
keep_check = ~isnan(raw.D);
mag = mag(:,repelem(keep_check, 2));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);

% decimal time in day of peaks
zulu_time = raw.peak_centre-floor(raw.peak_centre);
% full set split
keep_check = zulu_time > 0.333333333333 & zulu_time < 0.75  & raw.delta_P>-10;% & raw.D < 500;

mag = mag(:,repelem(keep_check, 2));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);
zulu_time = zulu_time(keep_check);

clear keep_check

%% day keep
keep_check = raw.Sol == 239;

mag = mag(:,repelem(keep_check, 2));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);
zulu_time = zulu_time(keep_check);

clear keep_check
%%
clear event_align
good_events = table2array(mag(:,1:6));
good_centres = raw.peak_centre(1:3);

%filter out 0s
good_events(good_events==0)=NaN;

%remove peak centre times
for i = 1:length(good_centres)
    good_events(:,1+2*(i-1)) = 88775.2440 .* (good_events(:,1+2*(i-1))- good_centres(i));
end
figure
hold on
for i = 1:length(good_centres)
    plot(good_events(:,1+2*(i-1)),good_events(:,2*i))
end
%linear detrend and resample to same times
interp_range = linspace(-100,100,400)';
for i = 1:length(good_centres)
    %linear detrend
    validIndices = ~isnan(good_events(:,1+2*(i-1))) & ~isnan(good_events(:,2*i));
    xValid = good_events(validIndices,1+2*(i-1));
    yValid = abs(good_events(validIndices,2*i));
    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = good_events(:,1+2*(i-1));
    fitted_y = feval(fitresult, fitted_x);

    good_events(:,2*i) = abs(good_events(:,2*i))-fitted_y;

    yValid = (good_events(validIndices,2*i));
    event_align(:,i) = interp1(xValid,yValid,interp_range,'linear','extrap');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
figure
hold on
for i = 1:length(good_centres)
    plot(good_events(:,1+2*(i-1)),good_events(:,2*i))
end

figure
hold on
for i = 1:length(good_centres)
    plot(interp_range,event_align(:,i))
end

mag_med = median((event_align),2);
plot(interp_range,mag_med,'k',LineWidth=2)
%%
filepath = '\ifg_data_calibrated\ifg_cal_SOL0239_2Hz_v06.tab';

mag_z = readtable(filepath,'FileType','text');
%%
mag_windows = [69266, 69666;
       75189, 75589;
       %%76447, 76847;
       %%78687, 79087;
       %%80411, 80811;
       80813, 81213;
       %%83242, 83642;
       %%86378, 86778;
       %%87468, 87868;
       %89373, 89773;
       %89636, 90036;
       %90603, 91003;
       %90782, 91182;
       %%94119, 94519;
       %96066, 96466;
       %%103586, 103986;
       %%104013, 104413;
       %105688, 106088;
       %%109451, 109851;
       %113420, 113820
       ];

mag_signal = [];
mag_times = [];
for i = 1:length(good_centres)
    background_end(i) = find(mag_z.MLST==table2array(mag(1,2*i-1)))+38;
    background_start(i) = background_end(i) - 440;
    background_mid(i) = (background_start(i) + background_end(i)) /2;
    mag_signal = [mag_signal mag_z.B_down(background_start(i):background_end(i))];
    mag_times = [mag_times 88775.244 *(mag_z.MLST(background_start(i):background_end(i))-mag_z.MLST(background_mid(i)))];
    mag_signal(:,i) = mag_signal(:,i) - median(mag_signal(:,i));
end

for i = 1:length(good_centres)
    %linear detrend
    [xData, yData] = prepareCurveData(mag_times(:,i), mag_signal(:,i));
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = mag_times(:,i);
    fitted_y = feval(fitresult, fitted_x);
  
    temp = yData - fitted_y;
    
    background_align(:,i) = interp1(xData,temp,interp_range,'linear','extrap');
    clear validIndices xData yData opts fitresult fitted_y fitted_x temp
end
figure
hold on
for i = 1:length(good_centres)
    plot(interp_range,background_align(:,i))
end
patch([interp_range' flip(interp_range)'],[1.58/sqrt(3) * iqr(abs(background_align),2)' flip(-1.58/sqrt(3) * iqr(abs(background_align),2))'],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7])


%%
