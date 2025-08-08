% Define the folders and instruments
folders = struct('ifg_data_calibrated', 'IFG', ...
                 'ps_data_calibrated', 'PS', ...
                 'sc_eng_data', 'ENG', ...
                 'seis_data_calibrated', 'SEIS', ...
                 'twins_data_derived', 'TWINS');

% Sample rate dictionary
rate_dict = containers.Map({'pt2Hz', '2Hz', '20Hz', '10Hz', 'gpt2Hz', '100Hz'}, ...
                           {0.2, 2.0, 20.0, 10.0, 0.2, 100.0});

% Initialize an empty array to store the information
data_list = [];

% Loop through each folder
folder_names = fieldnames(folders);
for i = 1:numel(folder_names)
    folder = folder_names{i};
    instrument = folders.(folder);
    files = dir(fullfile(folder, '*.csv'));
    files = [files; dir(fullfile(folder, '*.tab'))];  % Include .tab files
    
    for j = 1:numel(files)
        filename = files(j).name;
        filepath = fullfile(folder, filename);

        % Parse Sol and sample rate from the filename
        [sol, sample_rate_str] = parse_filename(filename, instrument);
        
        % Determine sample rate
        if isKey(rate_dict, sample_rate_str)
            sample_rate = rate_dict(sample_rate_str);
        else
            sample_rate = NaN;  % Default to NaN for unknown sample rates
            fid = fopen(filepath, 'r');
            fgetl(fid); % Skip the first row (header)
            second_row = fgetl(fid); % Read the second row
            

            data=regexp(second_row,',','split');
            data = cellfun(@(x) str2double(x), data);
            if strcmp(instrument, 'PS')
                sample_rate = data(7); % 7th column for PS instrument
            elseif strcmp(instrument, 'TWINS')
                if length(data)<9
                    second_row = fgetl(fid); %try next line
                    data = str2double(strsplit(second_row, ','));
                end
                sample_rate = data(9); % 9th column for TWINS instrument
            elseif strcmp(instrument,'ENG')
                sample_rate = 200;
            end
            fclose(fid);
        end
        
        % Append the data to the list (using a matrix for faster processing)
        data_list = [data_list; sol, i, sample_rate];
    end
end

% Convert data to a matrix
sol_values = unique(data_list(:, 1));
instrument_height = 1:numel(folder_names);
%%
% Plotting
figure;
hold on;

% Loop over each instrument
for i = 1:numel(folder_names)
    instrument_indices = data_list(:, 2) == i;
    sols_for_instrument = data_list(instrument_indices, 1);
    sample_rates = data_list(instrument_indices, 3);
    
    % Plot each unique sample rate for the instrument
    unique_sample_rates = unique(sample_rates);
    for j = 1:numel(unique_sample_rates)
        rate_indices = sample_rates == unique_sample_rates(j);
        y_offset = 0.1 * (j - 1); % Slight offset for each sample rate
        h= plot(sols_for_instrument(rate_indices), ...
             (i + y_offset) * ones(sum(rate_indices), 1), 'o', ...
             'MarkerFaceColor', 'auto', ... % Filled circles
             'DisplayName', sprintf('%s [%g Hz]', folders.(folder_names{i}), unique_sample_rates(j)));
        set(h, 'MarkerFaceColor', get(h, 'Color'));

    end
end

% Customize the plot
yticks(instrument_height);
yticklabels(struct2cell(folders));
xlabel('Sol');
ylabel('Instruments');
legend('Location', 'northeastoutside');  % Adjust legend position
grid on;
hold off;

% Helper Functions
function [sol, sample_rate] = parse_filename(filename, instrument)
    parts = strsplit(filename, '_');
    switch instrument
        case 'IFG'
            sol = str2double(parts{3}(4:end));
            sample_rate = parts{4};
        case 'PS'
            sol = str2double(parts{3}(4:7));
            sample_rate = 'Unknown';
        case 'ENG'
            sol = str2double(parts{2}(4:end));
            sample_rate = 'Unknown';
        case 'SEIS'
            sol = str2double(parts{3}(4:end));
            sample_rate = parts{4};
        case 'TWINS'
            sol = str2double(parts{3}(4:7));
            sample_rate = 'Unknown';
        otherwise
            error('Unknown instrument type');
    end
end

function nearest = find_nearest(array, value)
    [~, idx] = min(abs(array - value));
    nearest = array(idx);
end
