import os

files = []
rename = []
subdirs = [x[0] for x in os.walk('./')]

for subdir in subdirs:
    for file in os.listdir(subdir):
        if file.endswith('.csv') and file[0:9] == 'ps_calib_':
            files.append(os.path.join(subdir,file))
            rename.append(file[0:8]+'_SOL'+file[9:13]+'.csv')

for i in range(len(files)):
    try:
        os.rename(files[i], rename[i])
    except:
        try:
            os.rename(files[i], rename[i]+'1')
        except:
            os.rename(files[i], rename[i]+'2')
