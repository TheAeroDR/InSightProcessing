Sol = 101;
rate = 'pt2Hz';

filepath = '/ifg_data_calibrated/';
filepath = [filepath, 'ifg_cal_SOL', pad(num2str(Sol),4,'left','0'),'_',rate,'_v06.tab'];

filepath2 = '/sc_eng_data/';
filepath2 = [filepath2, 'ancil_SOL', pad(num2str(Sol),4,'left','0'),'_v01.tab'];
%%
mag_z = readtable(filepath,'FileType','text');
eng_data = readtable(filepath2,'FileType','text');
eng_time = convert_time(eng_data.SCET_UTC);
%%
figure
yyaxis left
plot(mag_z.MLST,mag_z.B_down)
hold on
for i = 1:10
plot([raw_real.peak_centre(i+651),raw_real.peak_centre(i+651)],[-905,-935],'--')
end
yyaxis right
%battery current
plot(eng_time(eng_data.E_0770~=9.9999e3),eng_data.E_0770(eng_data.E_0770~=9.9999e3))
hold on
%UHF op flag
plot(eng_time(eng_data.V_3531~=9.9999e3),eng_data.V_3531(eng_data.V_3531~=9.9999e3))
%x band
plot(eng_time(eng_data.G_0036~=9.9999e3),eng_data.G_0036(eng_data.G_0036~=9.9999e3))
%plot(a(eng_data.E_0114~=9.9999e3),eng_data.E_0114(eng_data.E_0114~=9.9999e3))
plot(eng_time(eng_data.E_0126~=9.9999e3),eng_data.E_0126(eng_data.E_0126~=9.9999e3))
%%
T_M_Quad_fig(eng_time(eng_data{:,53}~=9.9999e3), eng_data{:,53}(eng_data{:,53}~=9.9999e3),mag_z.MLST,mag_z.B_down, eng_data{:,54}(eng_data{:,54}~=9.9999e3), eng_data{:,60}(eng_data{:,60}~=9.9999e3), eng_data{:,61}(eng_data{:,61}~=9.9999e3));
%%
% figure
% tiledlayout(2,4)
% for i = 1:8
%     nexttile(i)
%     j = i+42;
%     yyaxis right
%     plot(mag_z.MLST,mag_z.B_down)
%     yyaxis left
%     hold on
%     plot(eng_time(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3));
%     legend([strrep(eng_data.Properties.VariableNames(j),'_','-'),'Data'])
% end
figure
yyaxis right
plot(mag_z.MLST,mag_z.B_down, 'DisplayName', 'Magentic Field')
yyaxis left
% j = 43;
% label = strrep(eng_data.Properties.VariableNames(j),'_','-');
% plot(eng_time(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3),'DisplayName',label{1});
% hold on
j = 47;
label = strrep(eng_data.Properties.VariableNames(j),'_','-');
plot(eng_time(eng_data{:,j}~=9.9999e3),eng_data{:,j}(eng_data{:,j}~=9.9999e3),'DisplayName',label{1});

legend()
%%
mag_time = mag_z.MLST(mag_z.MLST>239.5);
mag_ext = mag_z.B_down(mag_z.MLST>239.5);

mag_time = mag_time(mag_time<239.68);
mag_ext = mag_ext(mag_time<239.68);
%mag_time = mag_time(mag_time>239.32);
%mag_ext = mag_ext(mag_time>239.32);

[power,freq] = pspectrum(mag_ext,str2num(rate(1:end-2)));

figure
plot(log10(freq),log10(power))
hold on

[xData, yData] = prepareCurveData(log10(freq),log10(power));

ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [-1 0];

% Fit model to data.
fitresult = fit( xData, yData, ft);

fitted_D = linspace(-4,0, 100);
fitted_delta_P = feval(fitresult, fitted_D);
plot(fitted_D, fitted_delta_P);

%%
[a,b] = magic_spectrogram(mag_z,2,Sol);
%% loop through all days with real event to get the noise start and end times
%start_noise = [];
%end_noise = [];

[Sol,idx,~] = unique(raw.Sol);
Rate = raw.mag_rate(idx);

for i = 218:length(Sol)
    sol = Sol(i);
    90,577,579,587,600,686,687,699,701,
    
    rate = Rate(i);
    
    filepath_base = '/ifg_data_calibrated/';
    filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_',num2str(rate),'Hz_v06.tab'];
    filepath2_base = '/sc_eng_data/';
    filepath2 = [filepath2_base, 'ancil_SOL', pad(num2str(sol),4,'left','0'),'_v01.tab'];

    if rate == 0.2
           filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_pt2Hz_v06.tab'];
           if ~exist(filepath, 'file')
           filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_gpt2Hz_v06.tab'];
           end
    end
        
    mag_z = readtable(filepath,'FileType','text');
    eng = readtable(filepath2,'FileType','text');
    eng.MLST = convert_time(eng.SCET_UTC);
    [a,b] = magic_spectrogram(mag_z,rate,sol);
    
    yyaxis right
    plot(mag_z.MLST,mag_z.B_down,'Color',[0.6350 0.0780 0.1840])
    set(gca,'YColor',[0.6350 0.0780 0.1840]);

    j = 47;
    plot(eng.MLST(eng{:,j}~=9.9999e3),(50*((eng{:,j}(eng{:,j}~=9.9999e3)-28)./3.5))-940,'Color',[1.00,0.07,0.65],'LineStyle','-',Marker='.');

    fprintf('Current start_noise = %.4f, end_noise = %.4f\n', a, b);
    user_input = input('Do you want to adjust the noise start and end times? (y/n): ', 's');
    
    if strcmpi(user_input, 'y')
        a = input('Enter new start_noise value: ');
        b = input('Enter new end_noise value: ');
        
        % Update the plot with the new values
        hold on;
        xline(a, 'r', 'LineWidth', 2); % Update start line
        xline(b, 'g', 'LineWidth', 2); % Update end line
        hold off;
        
        disp('Press any key to continue...');
        pause;  % Wait for a key press
    end

    start_noise = [start_noise a];
    end_noise = [end_noise b];
end
%%
function dec_sol = convert_time(t_cell)
    a = split(t_cell,'T');
    b = split(a(:,1),'-');
    dt = datetime(cellfun(@(x) ['1-Jan-' x], b(:,1), 'UniformOutput', false)) + cellfun(@str2double, b(:,2)) - 1;
    c = split(a(:,2),':');
    t = duration(cellfun(@(x) [x{1} ':' x{2} ':' x{3}], num2cell(c, 2), 'UniformOutput', false),'InputFormat', 'hh:mm:ss.SSS');
    dec_sol = (dt + t) - (datetime(2018, 11, 26) + duration('05:10:50.33508','InputFormat', 'hh:mm:ss.SSSSS'));
    dec_sol = seconds(dec_sol) / 88775.2440;
end

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
    title({'-Y Solar Panel Temperature 1'},'LineWidth',1.5,'FontSize',30);
    xlim(axes1,[239 240]);
    grid(axes1,'on');
    hold(axes1,'off');
    set(axes1,'FontSize',30,'LineWidth',1.5);
    legend1 = legend(axes1,'show');
    set(legend1,'Position',[0.20138007970837 0.597291321171918 0.149739840583261 0.0942164200455395]);
    
    axes2 = nexttile(2);
    hold(axes2,'on');
    colororder([0.85 0.325 0.098]);
    yyaxis(axes2,'left');
    
    plot(X1,Y4,'DisplayName','T-0004','Parent',axes2,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes2,[-105 1]);
    set(axes2,'YColor',[0 0.447 0.741],'YTickLabel',{'','',''});  
    yyaxis(axes2,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes2,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylabel({'Magnetic Field [nT]'},'LineWidth',1.5,'FontSize',30);
    ylim(axes2,[-942 -899.6]);
    set(axes2,'YColor',[0.85 0.325 0.098]);
    title({'-Y Solar Panel Temperature 2'},'LineWidth',1.5,'FontSize',30);
    xlim(axes2,[239 240]);
    grid(axes2,'on');
    hold(axes2,'off');
    set(axes2,'FontSize',30,'LineWidth',1.5);
    legend2 = legend(axes2,'show');
    set(legend2,'Position',[0.202421746375036 0.121337755666113 0.149739840583261 0.0942164200455396]);
    
    axes3 = nexttile(3);
    hold(axes3,'on');
    colororder([0 0.447 0.741]);
    yyaxis(axes3,'left');

    plot(X1,Y3,'DisplayName','T-0019','Parent',axes3,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes3,[-105 1]);
    ylabel({'Temperature [deg C]'},'LineWidth',1.5,'FontSize',30);
    set(axes3,'YColor',[0 0.447 0.741]);
    yyaxis(axes3,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes3,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylim(axes3,[-942 -899.6]);
    set(axes3,'YColor',[0.85 0.325 0.098],'YTickLabel',{'','',''});
    title({'+Y Solar Panel Temperature 2'},'LineWidth',1.5,'FontSize',30);
    xlim(axes3,[239 240]);
    grid(axes3,'on');
    hold(axes3,'off');
    set(axes3,'FontSize',30,'LineWidth',1.5);
    legend3 = legend(axes3,'show');
    set(legend3,'Position',[0.664088413041704 0.121337755666113 0.149739840583261 0.0942164200455396]);
    
    axes4 = nexttile(4);
    hold(axes4,'on');
    colororder([0 0.447 0.741]);
    
    yyaxis(axes4,'left');
    plot(X1,Y5,'DisplayName','T-0018','Parent',axes4,'LineWidth',1.5,'Color',[0 0.447 0.741]);
    ylim(axes4,[-105 1]);
    set(axes4,'YColor',[0 0.447 0.741],'YTickLabel',{'','',''});
    yyaxis(axes4,'right');
    plot(X2,Y2,'DisplayName','Magnetic Field','Parent',axes4,'LineWidth',1.5,'Color',[0.85 0.325 0.098]);
    ylabel({'Magnetic Field [nT]'},'LineWidth',1.5,'FontSize',30);
    ylim(axes4,[-942 -899.6]);
    set(axes4,'YColor',[0.85 0.325 0.098]);
    title({'+Y Solar Panel Temperature 1'},'LineWidth',1.5,'FontSize',30);
    xlim(axes4,[239 240]);
    grid(axes4,'on');
    hold(axes4,'off');
    set(axes4,'FontSize',30,'LineWidth',1.5);
    legend4 = legend(axes4,'show');
    set(legend4,'Position',[0.66513007970837 0.597291321171918 0.149739840583261 0.0942164200455395]);
end

function [noise_start, noise_end] = magic_spectrogram(mag_z, rate, Sol)
    window = 1024;
    noverlap = 512;
    nfft = 1024;
    fs = rate;

    figure(1);
    clf;
    
    [S,F,T,P] = spectrogram(mag_z.B_down, window, noverlap, nfft, fs, 'yaxis', 'power');
    
    imagesc((T/88775.244 + Sol), F, 10*log10(P));

    set(gca,'YDir','normal')
    c = colorbar;
    c.Label.String = 'Power [dB]';

    ylabel("Frequency [Hz]")

    power_threshold = 0.00005;
    
    broadband_freq_range = [0,1];
    
    freq_indices = find(F >= broadband_freq_range(1) & F <= broadband_freq_range(2));
    
    broadband_presence = sum(P(freq_indices, :) > power_threshold, 1) > (0.5 * length(freq_indices)); % Example: 50% of broadband frequencies
    
    broadband_start = find(diff([0 broadband_presence]) == 1);
    broadband_stop = find(diff([broadband_presence 0]) == -1);
    
    broadband_start_times = T(broadband_start)/88775.2440 + Sol;
    broadband_stop_times = T(broadband_stop)/88775.2440 + Sol;
    
    hold on;
    
    xline(broadband_start_times(1), 'r', 'LineWidth', 2); % Red line for start times
    xline(broadband_stop_times(end), 'g', 'LineWidth', 2); % Green line for stop times
    
    noise_start = broadband_start_times(1);
    noise_end = broadband_stop_times(end);
end