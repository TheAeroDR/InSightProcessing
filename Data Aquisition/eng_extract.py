import os

files = []
rename = []
subdirs = [x[0] for x in os.walk('./')]

for subdir in subdirs:
    for file in os.listdir(subdir):
        if file.endswith('.tab'):
            files.append(os.path.join(subdir,file))
            rename.append((file[0:13] + '_v01.tab') )

for i in range(len(files)):
    os.rename(files[i], rename[i])