import os

files = []
files_20 = []
rename_20 = []
files_unsort = []
files_2 = []
rename_2 = []
files_pt2 = []
rename_pt2 = []
files_gpt2 = []
rename_gpt2 = []
files_10 = []
rename_10 = []
subdirs = [x[0] for x in os.walk('./')]

for subdir in subdirs:
    for file in os.listdir(subdir):
        if file.endswith('.tab'):
            files.append(os.path.join(subdir,file))
            if file.endswith('_20Hz_v06.tab'):
                files_20.append(os.path.join(subdir,file))
                rename_20.append((file[0:15] + '_20Hz_v06.tab') )
            elif file.endswith('_10Hz_v06.tab'):
                files_10.append(os.path.join(subdir,file))
                rename_10.append((file[0:15] + '_10Hz_v06.tab') )
            elif file.endswith('_2Hz_v06.tab'):
                files_2.append(os.path.join(subdir,file))
                rename_2.append((file[0:15] + '_2Hz_v06.tab') )
            elif file.endswith('_pt2Hz_v06.tab'):
                files_pt2.append(os.path.join(subdir,file))
                rename_pt2.append((file[0:15] + '_pt2Hz_v06.tab') )
            elif file.endswith('_gpt2Hz_v06.tab'):
                files_gpt2.append(os.path.join(subdir,file))
                rename_gpt2.append((file[0:15] + '_gpt2Hz_v06.tab') )
            else:
                files_unsort.append(os.path.join(subdir,file))

for i in range(len(files_10)):
    os.rename(files_10[i], rename_10[i])
for i in range(len(files_20)):
    os.rename(files_20[i], rename_20[i])
for i in range(len(files_2)):
    try:
        os.rename(files_2[i], rename_2[i])
    except:
        os.rename(files_2[i], rename_2[i] + '1')
    for i in range(len(files_pt2)):
        os.rename(files_pt2[i], rename_pt2[i])
    for i in range(len(files_gpt2)):
        try:
            os.rename(files_gpt2[i], rename_gpt2[i])
        except:
            os.rename(files_gpt2[i], rename_gpt2[i] + '1')