import pandas as pd
import os
import shutil
import logging
from Insight_Module import Sol
#from Insight_pressure_only import Sol
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logfile.txt",
    filemode="a"
    )

def run_sol(i,filestring):
    key_data, mag_out, pres_out, mag_bcgnd = Sol(i,0)
    a=1
    if not key_data.empty:
        key_data.to_csv(f"{filestring}_key.csv")
    #if not mag_out.empty:
    #    mag_out.to_csv(f"{filestring}_mag.csv")
    #if not pres_out.empty:
    #    pres_out.to_csv(f"{filestring}_pres.csv")
    #if not mag_bcgnd.empty:
    #    mag_bcgnd.to_csv(f"{filestring}_background.csv")
 
if __name__ == '__main__':
    print(time.gmtime())
    os.makedirs('temp', exist_ok=True)
    start_sol = 75 
    end_sol = 1250
    filenames = [f"./temp/output_{i}" for i in range(start_sol,end_sol)]
    for i, filename in zip(range(start_sol,end_sol), filenames):
        run_sol(i, filename)
        
    concat_output = pd.concat([pd.read_csv(f"{filename}_key.csv") for filename in filenames if os.path.exists(f"{filename}_key.csv")])
    concat_output.to_csv('/media/david/Extreme SSD/dataout_test5.csv')

    #mag_output = pd.concat([pd.read_csv(f"{filename}_mag.csv") for filename in filenames if os.path.exists(f"{filename}_mag.csv")],axis=1)
    #mag_output.to_csv('/media/david/Extreme SSD/magout_1000s.csv',index=False)

    #pres_output = pd.concat([pd.read_csv(f"{filename}_pres.csv") for filename in filenames if os.path.exists(f"{filename}_pres.csv")],axis=1)
    #pres_output.to_csv('/media/david/Extreme SSD/presout_1000s.csv',index=False)

    #bckgnd_output = pd.concat([pd.read_csv(f"{filename}_background.csv") for filename in filenames if os.path.exists(f"{filename}_background.csv")],axis=1)
    #bckgnd_output.to_csv('/media/david/Extreme SSD/backgroundout_1000s.csv',index=False)

    shutil.rmtree("./temp/")
    print(time.gmtime())