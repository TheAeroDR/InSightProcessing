%raw = readtable('dataout.csv');
%mag = readtable('magout.csv');
%pres = readtable('presout.csv');
%%
load("raw.mat")
%%
% filtering
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

% decimal time in day of peaks
zulu_time = raw.peak_centre-floor(raw.peak_centre);
% full set split
keep_check = zulu_time > 8/24 & zulu_time < 17/24;

mag = mag(:,repelem(keep_check, 4));
pres = pres(:,repelem(keep_check, 2));
mag_bckgnd = mag_bckgnd(:,repelem(keep_check, 4));
raw = raw(keep_check, :);
zulu_time = zulu_time(keep_check);

clear keep_check

keep_check = raw.Sol == 239;

mag = mag(:,repelem(keep_check, 4));
mag_bckgnd = mag_bckgnd(:,repelem(keep_check, 4));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);
zulu_time = zulu_time(keep_check);

clear keep_check

%%
magz = readtable('ifg_cal_SOL0239_20Hz_v06.tab','FileType','text');
ifg_2 = readtable('ifg_cal_SOL0239_2Hz_v06.tab','FileType','text');
ifg_p2 = readtable('ifg_cal_SOL0239_gpt2Hz_v06.tab','FileType','text');

file_sol = 239;

h = magz.TLST;
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/24;
magz.TLST = decimal_sol_ifg;

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
axZ = plot(magz.TLST, magz.B_down);
hold on
set(gca,'xticklabel',[])
xlim(gca,[239,240])
ylabel('$B_Z$ [nT]')

nexttile(2)
axN = plot(magz.TLST, magz.B_north);
hold on
set(gca,'xticklabel',[])
ylabel('$B_N$ [nT]')
xlim(gca,[239,240])

nexttile(3)
axE = plot(magz.TLST, magz.B_east);
hold on
xlim(gca,[239,240])
ylabel('$B_E$ [nT]')
xlabel('Decimal Sol [-]')

colours = lines(3);

set(axZ, 'Color', colours(1,:));
set(axN, 'Color', colours(2,:));
set(axE, 'Color', colours(3,:));

lengths = [-905,-935;-1320,-1370;1175,1150];
noise_times = [239.5 239.685];
for i =1:3
    nexttile(i)
    plot([raw.peak_centre raw.peak_centre],[lengths(i,1) lengths(i,2)])
    ylim([lengths(i,2),lengths(i,1)])
    patch([noise_times(1) noise_times(1) noise_times(2) noise_times(2)],[lengths(i,2), lengths(i,1), lengths(i,1), lengths(i,2)],'m','FaceAlpha',0.5)
end

%%
[~,idx1] = min(abs(magz.TLST - noise_times(1)));
noise_win(1) = idx1;
[~,idx2] = min(abs(magz.TLST - noise_times(2)));
noise_win(2) = idx2;

noisy_mag{1,1} = [magz.B_down(noise_win(1):noise_win(2)), magz.B_north(noise_win(1):noise_win(2)), magz.B_east(noise_win(1):noise_win(2))];

noisy_mag{2,1} = resample(noisy_mag{1,1},1,10);
noisy_mag{3,1} = resample(noisy_mag{1,1},1,100);

[~,idx1] = min(abs(ifg_2.TLST - noise_times(1)));
noise_win(1) = idx1;
[~,idx2] = min(abs(ifg_2.TLST - noise_times(2)));
noise_win(2) = idx2;

noisy_mag{1,2} = [ifg_2.B_down(noise_win(1):noise_win(2)), ifg_2.B_north(noise_win(1):noise_win(2)), ifg_2.B_east(noise_win(1):noise_win(2))];

noisy_mag{2,2} = resample(noisy_mag{1,2},1,10);

[~,idx1] = min(abs(ifg_p2.TLST - noise_times(1)));
noise_win(1) = idx1;
[~,idx2] = min(abs(ifg_p2.TLST - noise_times(2)));
noise_win(2) = idx2;

noisy_mag{1,3} = [ifg_p2.B_down(noise_win(1):noise_win(2)), ifg_p2.B_north(noise_win(1):noise_win(2)), ifg_p2.B_east(noise_win(1):noise_win(2))];

sf = [20,2,0.2];
lim = [-1.3,-2.4,-3.4,-2.3,-3.4,-3.35;0.2,-0.2,-1.1,-0.2,-1.04,-1.04;1,0,-1,0,-1,-1]';
titles={'20Hz','2Hz','0.2Hz','20Hz to 2Hz','2 to 0.2Hz','20Hz to 0.2Hz'};
figure
tiledlayout(3,1)
for i = 1:3
    [p,f] = pspectrum(noisy_mag{1,i},sf(i),"power");
    nexttile(i)
    plot(log10(f),log10(p))
    hold on
    
    [xData, yData] = prepareCurveData(log10(f),log10(p(:,1)));
    
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    
    % Fit model to data.
    [fitresult,gof] = fit(xData(xData>lim(i,1)&xData<lim(i,2)), yData(xData>lim(i,1)&xData<lim(i,2)), ft);
    fitted_D = linspace(lim(i,1),lim(i,2), 100);
    fitted_delta_P = feval(fitresult, fitted_D);
    plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
    fitresult.p1

    [fitresult,gof] = fit(xData(xData>lim(i,2)&xData<lim(i,3)), yData(xData>lim(i,2)&xData<lim(i,3)), ft);
    fitted_D = linspace(lim(i,2),lim(i,3), 100);
    fitted_delta_P = feval(fitresult, fitted_D);
    plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
    fitresult.p1

    xlim([-5,1])
    title(titles{i})
    if i ==1 || i ==2
        set(gca, 'XTickLabel', [])
    end
    if i ==3
        xlabel('$\log_{10}(\textrm{Frequency})$ [Hz]')
    end
    if i ==2
        ylabel('$\log_{10}(\textrm{Power})$ [nT\textsuperscript{2}/Hz]')
    end
end

figure
tiledlayout(2,1)
for i = 1:2
    [p,f] = pspectrum(noisy_mag{2,i},sf(i+1),"power");
    nexttile(i)
    plot(log10(f),log10(p))
    hold on
    
    [xData, yData] = prepareCurveData(log10(f),log10(p(:,1)));
    
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    
    % Fit model to data.
    [fitresult,gof] = fit(xData(xData>lim(i+3,1)&xData<lim(i+3,2)), yData(xData>lim(i+3,1)&xData<lim(i+3,2)), ft);
    fitted_D = linspace(lim(i+3,1),lim(i+3,2), 100);
    fitted_delta_P = feval(fitresult, fitted_D);
    plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
    fitresult.p1

    [fitresult,gof] = fit(xData(xData>lim(i+3,2)&xData<lim(i+3,3)), yData(xData>lim(i+3,2)&xData<lim(i+3,3)), ft);
    fitted_D = linspace(lim(i+3,2),lim(i+3,3), 100);
    fitted_delta_P = feval(fitresult, fitted_D);
    plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
    fitresult.p1

    xlim([-5,1])
    title(titles{i+3})
    if i ==1
        set(gca, 'XTickLabel', [])
    end
    if i ==3
        xlabel('$\log_{10}(\textrm{Frequency})$ [Hz]')
    end
    if i ==2
        ylabel('$\log_{10}(\textrm{Power})$ [nT\textsuperscript{2}/Hz]')
    end
end

figure
[p,f] = pspectrum(noisy_mag{3,1},sf(3),"power");
nexttile(i)
plot(log10(f),log10(p))
hold on

[xData, yData] = prepareCurveData(log10(f),log10(p(:,1)));

ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [-1 0];

% Fit model to data.
[fitresult,gof] = fit(xData(xData>lim(6,1)&xData<lim(6,2)), yData(xData>lim(6,1)&xData<lim(6,2)), ft);
fitted_D = linspace(lim(6,1),lim(6,2), 100);
fitted_delta_P = feval(fitresult, fitted_D);
plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
fitresult.p1

[fitresult,gof] = fit(xData(xData>lim(6,2)&xData<lim(6,3)), yData(xData>lim(6,2)&xData<lim(6,3)), ft);
fitted_D = linspace(lim(6,2),lim(6,3), 100);
fitted_delta_P = feval(fitresult, fitted_D);
plot(fitted_D, fitted_delta_P,'k',LineWidth=2);
fitresult.p1

xlim([-5,1])
title(titles{6})
xlabel('$\log_{10}(\textrm{Frequency})$ [Hz]')
ylabel('$\log_{10}(\textrm{Power})$ [nT\textsuperscript{2}/Hz]')

%%
eng_data = readtable("ancil_SOL0239_v01.tab",'FileType','text');

h = seconds(eng_data.LTST);
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
if cumsum(wrap(1:length(wrap)/2)) ==0
    wrap(1)=true;
end
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/86400;
eng_data.LTST = decimal_sol_ifg;

%%
T_M_Quad_fig(eng_data.LTST(eng_data{:,53}~=9.9999e3), eng_data{:,53}(eng_data{:,53}~=9.9999e3),magz.TLST,magz.B_down, eng_data{:,54}(eng_data{:,54}~=9.9999e3), eng_data{:,60}(eng_data{:,60}~=9.9999e3), eng_data{:,61}(eng_data{:,61}~=9.9999e3));

%%
figure
yyaxis right
plot(magz.TLST,magz.B_down, 'DisplayName', 'Magnetic Field')
hold on
yyaxis left
% j = 43;
% label = strrep(eng_data.Properties.VariableNames(j),'_','-');
% plot(eng_time(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3),'DisplayName',label{1});
% hold on
j = 47;
label = strrep(eng_data.Properties.VariableNames(j),'_','-');
plot(eng_data.LTST(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3),'DisplayName',label{1},'Marker',"^");
legend()
ylabel('Voltage [V]')
yyaxis right
plot([noise_times(1) noise_times(1)],[lengths(1,1) lengths(1,2)],'-g')
plot([noise_times(2) noise_times(2)],[lengths(1,1) lengths(1,2)],'-m')
ylabel('$B_Z$[nT]')
xlabel('Decimal Sol [-]')
%%
magz_330 = readtable("ifg_cal_SOL0330_2Hz_v06.tab","FileType","text");

file_sol = 330;

h = magz_330.TLST;
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
if cumsum(wrap(1:length(wrap)/2)) ==0
    wrap(1)=true;
end
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/24;
magz_330.TLST = decimal_sol_ifg;

eng_data_330 = readtable("ancil_SOL0330_v01.tab",'FileType','text');

h = seconds(eng_data_330.LTST);
solflag = zeros(size(h));
wrap = [false; diff(h) < 0];
if cumsum(wrap(1:length(wrap)/2)) ==0
    wrap(1)=true;
end
solflag = solflag + cumsum(wrap);
decimal_sol_ifg = (file_sol - 1) + solflag + h/86400;
eng_data_330.LTST = decimal_sol_ifg;

noise_times = [330.517 330.657];
lengths(1,1) = -918;
lengths(1,2) = -939;
%%
figure
yyaxis right
plot(magz_330.TLST,magz_330.B_down, 'DisplayName', 'Magnetic Field')
hold on
yyaxis left
% j = 43;
% label = strrep(eng_data.Properties.VariableNames(j),'_','-');
% plot(eng_time(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3),'DisplayName',label{1});
% hold on
j = 47;
label = strrep(eng_data_330.Properties.VariableNames(j),'_','-');
plot(eng_data_330.LTST(eng_data_330{:,j}~=9.9999e3),eng_data_330{:,j}(eng_data_330{:,j}~=9.9999e3),'DisplayName',label{1},'Marker',"^");
legend()
ylabel('Voltage [V]')
yyaxis right
plot([noise_times(1) noise_times(1)],[lengths(1,1) lengths(1,2)],'-g')
plot([noise_times(2) noise_times(2)],[lengths(1,1) lengths(1,2)],'-m')
ylabel('$B_Z$[nT]')
xlabel('Decimal Sol [-]')
%%
lengths = [-905,-935;-1320,-1370;1175,1150];
noise_times = [239.5 239.685];
%%
clear event_align
keep_check = raw.peak_centre<noise_times(1) | raw.peak_centre>noise_times(2);

mag = mag(:,repelem(keep_check, 4));
mag_bckgnd = mag_bckgnd(:,repelem(keep_check, 4));
pres = pres(:,repelem(keep_check, 2));
raw = raw(keep_check, :);

good_events = table2array(mag);
good_centres = raw.peak_centre;

%filter out 0s
%good_events(good_events==0)=NaN;

%remove peak centre times
for i = 1:length(good_centres)
    good_events(:,1+4*(i-1)) = (87755.244).* (good_events(:,1+4*(i-1))- good_centres(i));
end
figure
hold on
for i = 1:length(good_centres)
    plot(good_events(:,1+4*(i-1)),good_events(:,4*i))
end
%linear detrend and resample to same times
interp_range = linspace(-400,400,1000)';
pol_win = 10;
for i = 1:length(good_centres)
    %linear detrend
    validIndices = ~isnan(good_events(:,1+4*(i-1))) & ~isnan(good_events(:,4*i));
    xValid = good_events(validIndices,1+4*(i-1));
    yValid = good_events(validIndices,4*i);

    [xData, yData] = prepareCurveData(xValid, yValid);
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [-1 0];
    fitresult = fit(xData, yData, ft);  
    fitted_x = good_events(:,1+4*(i-1));
    fitted_y = feval(fitresult, fitted_x);

    good_events(:,4*i) = (good_events(:,4*i))-fitted_y;

    yValid = (good_events(validIndices,4*i));
    event_align(:,i) = interp1(xValid,yValid,interp_range,'linear');

    clear validIndices xValid yValid xData yData opts fitresult fitted_y fitted_x
end
figure
hold on
for i = 1:length(good_centres)
    plot(good_events(:,1+4*(i-1)),good_events(:,4*i))
end

pol_idx = abs(interp_range)<pol_win;
for i = 1:length(good_centres)
    polScore = median(event_align(pol_idx,i), 'omitnan');
    pol(i) = sign(polScore);
end

colours = lines(7);
colours = [colours 0.2*ones(7,1)];
figure
hold on
for i = 1:length(good_centres)
    plot(interp_range,pol(i) * event_align(:,i), Color=colours(i,:))
end

mag_med = median((pol(i) * event_align),2);
plot(interp_range,mag_med,'k',LineWidth=2)
xlabel('Time [s]')
ylabel('$B_Z$ [nT]')
%%
clear mag_signal mag_times
for i = 1:length(good_centres)
    mag_signal(:,i) = [mag_bckgnd(:,4*i)];
    mag_times(:,i) = [mag_bckgnd(:,4*i - 3)];
    mag_signal(:,i) = mag_signal(:,i) - median(mag_signal(:,i));
    mag_times(:,i) = (86400) .* (mag_times(:,i) - median(mag_times(:,i)));
end

mag_signal = table2array(mag_signal);
mag_times = table2array(mag_times);

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
    
    background_align(:,i) = pol(i) * interp1(xData,temp,interp_range,'linear','extrap');
    clear validIndices xData yData opts fitresult fitted_y fitted_x temp
end
figure
hold on
for i = 1:length(good_centres)
    plot(interp_range,background_align(:,i))
end
patch([interp_range' flip(interp_range)'],[1.58/sqrt(3) * iqr(abs(background_align),2)' flip(-1.58/sqrt(3) * iqr(abs(background_align),2))'],[.7 .7 .7],'facealpha',0.3,'edgecolor',[.7 .7 .7])

%%
[a,b] = magic_spectrogram(magz,eng_data,20);
%%
function T_M_Quad_fig(X1, Y1, X2, Y2, Y3, Y4, Y5)
    figure
    tiledlayout(2,2);
    
    axes1 = nexttile(1);
    hold(axes1,'on');
    
    yyaxis(axes1,'left');
    plot(X1,Y1,'DisplayName','T-0003','Parent',axes1,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylabel({'Temperature [deg C]'},'LineWidth',1.5,'FontSize',30);
    ylim(axes1,[-105 1]);
    set(axes1,'YColor',[0 0.447 0.741]);
    yyaxis(axes1,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes1,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylim(axes1,[-942 -899.6]);
    set(axes1,'YColor',[0.85 0.325 0.098],'YTickLabel',{'','',''});
    xlim(axes1,[239 240]);
    grid(axes1,'on');
    hold(axes1,'off');
    set(axes1,'FontSize',30,'LineWidth',1.5);
    legend1 = legend(axes1,'show');
    set(legend1,'Position',[0.20138007970837 0.597291321171918 0.149739840583261 0.0942164200455395]);
    uistack(legend1, 'top');

    axes2 = nexttile(2);
    hold(axes2,'on');
    colororder([0.85 0.325 0.098]);
    yyaxis(axes2,'left');
    
    plot(X1,Y4,'DisplayName','T-0018','Parent',axes2,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes2,[-105 1]);
    set(axes2,'YColor',[0 0.447 0.741],'YTickLabel',{'','',''});  
    yyaxis(axes2,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes2,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylabel({'Magnetic Field [nT]'},'LineWidth',1.5,'FontSize',30);
    ylim(axes2,[-942 -899.6]);
    set(axes2,'YColor',[0.85 0.325 0.098]);
    xlim(axes2,[239 240]);
    grid(axes2,'on');
    hold(axes2,'off');
    set(axes2,'FontSize',30,'LineWidth',1.5);
    legend2 = legend(axes2,'show');
    set(legend2,'Position',[0.202421746375036 0.121337755666113 0.149739840583261 0.0942164200455396]);
    uistack(legend2, 'top');

    axes3 = nexttile(3);
    hold(axes3,'on');
    colororder([0 0.447 0.741]);
    yyaxis(axes3,'left');

    plot(X1,Y3,'DisplayName','T-0004','Parent',axes3,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes3,[-105 1]);
    ylabel({'Temperature [deg C]'},'LineWidth',1.5,'FontSize',30);
    set(axes3,'YColor',[0 0.447 0.741]);
    yyaxis(axes3,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes3,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylim(axes3,[-942 -899.6]);
    set(axes3,'YColor',[0.85 0.325 0.098],'YTickLabel',{'','',''});
    xlim(axes3,[239 240]);
    grid(axes3,'on');
    hold(axes3,'off');
    set(axes3,'FontSize',30,'LineWidth',1.5);
    legend3 = legend(axes3,'show');
    set(legend3,'Position',[0.664088413041704 0.121337755666113 0.149739840583261 0.0942164200455396]);
    uistack(legend3, 'top');

    axes4 = nexttile(4);
    hold(axes4,'on');
    colororder([0 0.447 0.741]);
    
    yyaxis(axes4,'left');
    plot(X1,Y5,'DisplayName','T-0019','Parent',axes4,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes4,[-105 1]);
    set(axes4,'YColor',[0 0.447 0.741],'YTickLabel',{'','',''});
    yyaxis(axes4,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes4,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylabel({'Magnetic Field [nT]'},'LineWidth',1.5,'FontSize',30);
    ylim(axes4,[-942 -899.6]);
    set(axes4,'YColor',[0.85 0.325 0.098]);
    xlim(axes4,[239 240]);
    grid(axes4,'on');
    hold(axes4,'off');
    set(axes4,'FontSize',30,'LineWidth',1.5);
    legend4 = legend(axes4,'show');
    set(legend4,'Position',[0.66513007970837 0.597291321171918 0.149739840583261 0.0942164200455395]);
    set(axes4, 'Layer', 'top');
end

function [noise_start, noise_end] = magic_spectrogram(mag_z,eng, rate)
    window = 256;
    noverlap = 128;
    nfft = 256;
    fs = rate;

    figure(1);
    clf;
    
    [~,F,T,P] = spectrogram(mag_z.B_down, window, noverlap, nfft, fs, 'yaxis', 'power');
    
    imagesc((T/88775.244 + mag_z.TLST(1)), F, 10*log10(P));

    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.String = 'Power [dB]';

    ylabel("Frequency [Hz]")
    xlabel("Decimal Sol [-]")

    power_threshold = 0.00065;
    
    broadband_freq_range = [0,fs/2];
    
    freq_indices = find(F >= broadband_freq_range(1) & F <= broadband_freq_range(2));
    
    broadband_presence = sum(P(freq_indices, :) > power_threshold, 1) > (0.5 * length(freq_indices));
    
    broadband_start = find(diff([0 broadband_presence]) == 1);
    broadband_stop = find(diff([broadband_presence 0]) == -1);
    
    broadband_start_times = T(broadband_start)/88775.244 + mag_z.TLST(1);
    broadband_stop_times = T(broadband_stop)/88775.244 + mag_z.TLST(1);
    
    hold on;
    
    xline(broadband_start_times(1), 'r', 'LineWidth', 2); % Red line for start times
    xline(broadband_stop_times(end), 'g', 'LineWidth', 2); % Green line for stop times
    
    noise_start = broadband_start_times(1);
    noise_end = broadband_stop_times(end);

    yyaxis right
    plot(mag_z.TLST,mag_z.B_down,'Color',[0.6350 0.0780 0.1840])
    set(gca,'YColor',[0.6350 0.0780 0.1840]);
    yticklabels([])

    j = 47;
    plot(eng.LTST(eng{:,j}~=9.9999e3),(50*((eng{:,j}(eng{:,j}~=9.9999e3)-28)./3.5))-940,'Color',[1.00,0.07,0.65],'LineStyle','-',Marker='.');

    end