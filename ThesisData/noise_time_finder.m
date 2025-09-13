%raw = readtable('dataout_test3.csv');

%[Sol,idx,~] = unique(raw.Sol);
%Rate = raw.mag_rate(idx);

for i = 235:length(Sol)
    sol = Sol(i);
    
    rate = Rate(i);
    
    filepath_base = './ifg_data_calibrated/';
    filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_',num2str(rate),'Hz_v06.tab'];
    filepath2_base = './sc_eng_data_LTST/';
    filepath2 = [filepath2_base, 'ancil_SOL', pad(num2str(sol),4,'left','0'),'_v01.tab'];

    if rate == 0.2
       filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_pt2Hz_v06.tab'];
       if ~exist(filepath, 'file')
            filepath = [filepath_base, 'ifg_cal_SOL', pad(num2str(sol),4,'left','0'),'_gpt2Hz_v06.tab'];
       end
    end
        
    mag_z = readtable(filepath,'FileType','text');
    eng = readtable(filepath2,'FileType','text');

    h = mag_z.TLST;
    solflag = zeros(size(h));
    wrap = [false; diff(h) < 0];
    solflag = solflag + cumsum(wrap);
    decimal_sol_ifg = (sol - 1) + solflag + h/24;
    mag_z.TLST = decimal_sol_ifg;

    h = seconds(eng.LTST);
    solflag = zeros(size(h));
    wrap = [false; diff(h) < 0];
    if cumsum(wrap(1:length(wrap)/2)) ==0
        wrap(1)=true;
    end
    solflag = solflag + cumsum(wrap);
    decimal_sol_ifg = (sol - 1) + solflag + h/86400;
    eng.LTST = decimal_sol_ifg;

    [a,b] = magic_spectrogram(mag_z,eng,rate);
    
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

    start_noise(i) = a;
    end_noise(i) = b;
end
%%


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