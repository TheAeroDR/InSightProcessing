%read data from file, specifiy delimiter as comma, and that the top row
%contains dat rather than variable names
data = readtable("arm_activity_LTST.txt",'Delimiter',',','ReadVariableNames',false);

%extract the first two columns of data, for arm there is only one mode,
%multiple modes of different colours can be added easily
data = data(:,[2,4]);

%convert table to cell array to allow operation on the date/time strings
data = table2cell(data);

%custom function to convert UTC to decimal Sol
sol = decimalise(data);

%create figure
figure
hold on

%for each pair of values make a patch (a filled rectangle) of arbitrary height
for i = 1:length(sol)
    patch([sol(i,1),sol(i,1),sol(i,2), sol(i,2)],[100,-100,-100,100],'k') %formerly had ",LineWidth',3" after the k and before the end bracket
end
%x axis label
xlabel('Sol')
%turn off y ticks and y tick labels
yticklabels([])
yticks([])

%%
mag = readtable("ifg_cal_SOL0253_2Hz_v06.tab",'FileType','text');

mag.TLST = mag.TLST/24 + 253;

plot(mag.TLST,mag.B_down)
hold on

for i = 1:length(sol)
    if floor(sol(i,1)) == 253
        patch([sol(i,1),sol(i,1),sol(i,2), sol(i,2)],[-950,-900,-900,-950],'k','FaceAlpha',0.5)
    end
end

%% custom function to convert array of UTC times to array of decimal Sol
%times for InSight
function d = decimalise(ltst)

    %define seconds per day LTST
    sec_per_day = 86400;
    
    %split string at " " into sol string and time string
    [date_str, time_str] = strtok(ltst, ' ');
    %split hour
    [HH,rem3] = strtok(time_str,':');
    %split minute
    [mm,ss] = strtok(rem3,':');
    %split second
    [ss,~]=strtok(ss,':');
    
    %convert strings to doubles
    sol = str2double(date_str);
    HH = str2double(HH);
    mm = str2double(mm);
    ss = str2double(ss);
    
    LTST_s = ((HH * 3600) + (mm * 60) + ss)./sec_per_day;

    d = sol + LTST_s;
end
