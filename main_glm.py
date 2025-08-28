# %%
import numpy as np
import os
from os.path import join, exists, split
import time
from pathlib import Path
import sys
from glmsingle.glmsingle import GLM_single
import nibabel as nib
import psutil  # For memory monitoring
import gc

# %%
homedir = split(os.getcwd())[0]
outputdir_glmsingle = join(homedir,'thesis_code/GLMOutputs2-sub04-ses02')

# %%
base_path = '/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
sub = '04'
ses = '2'
run = '1'

data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg_2mm.nii.gz'
data_path_org = join(base_path, f'sub-pd0{sub}',f'ses-{ses}','func', data_name)
data_run1 = nib.load(data_path_org).get_fdata()

run = '2'
data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg_2mm.nii.gz'
data_path_org = join(base_path, f'sub-pd0{sub}',f'ses-{ses}','func', data_name)
data_run2 = nib.load(data_path_org).get_fdata()

X = {}
X['data'] = data_run1
data = []
data.append(data_run1) #(180, 256, 170, 850)
data.append(data_run2)

# %%
fs = 1000 
TR = 1.0  
T = 850    
num_trials = 9 * 10  

xyz = data_run1.shape[:3]
xyzt = data_run1.shape
stimdur = 9 #<stimdur> is the duration of a trial in seconds. For example, 3.5 means that you expect the neural activity from a given trial to last for 3.5 s.
tr = 1


# %%
path = f'/Volumes/McKeownLab/Data_Masterfile/Zahra-Thesis-Data/Go-times/PSPD0{sub}-ses-{ses}-go-times.txt'
go_flag = np.loadtxt(path, dtype=int)
go_flag1 = go_flag[0,:]
go_flag2 = go_flag[1,:]

# %%
design_matrix = []
# time_count = 0

design1 = np.zeros((T, num_trials), dtype=int)
for i in range(num_trials):
    # go_flag = round(flag_time[i]) + time_count
    # print(go_flag)
    design1[go_flag1[i]-1, i] = 1

design_matrix.append(design1)

design2 = np.zeros((T, num_trials), dtype=int)
for i in range(num_trials):
    # go_flag = round(flag_time[i]) + time_count
    # print(go_flag)
    design2[go_flag2[i]-1, i] = 1
design_matrix.append(design2)

opt = dict()

# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# MEMORY OPTIMIZATION: Reduce chunk size to prevent memory exhaustion
opt['chunklen'] = 5000  # Default is 50000, reducing to 5000

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [0,0,0,1]  # Only keep final model in memory to save RAM

glmsingle_obj = GLM_single(opt)
start_time = time.time()

if not exists(outputdir_glmsingle):
    print(f'running GLMsingle...')

    # Monitor memory usage
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f'Memory usage before GLMsingle: {memory_before:.2f} GB')

    # Force garbage collection before starting
    gc.collect()

    # run GLMsingle
    results_glmsingle = glmsingle_obj.fit(design_matrix,data,stimdur,tr,outputdir=outputdir_glmsingle, figuredir=outputdir_glmsingle)
else:
    print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir_glmsingle}')
    # load existing file outputs if they exist
    results_glmsingle = dict()
    results_glmsingle['typea'] = np.load(join(outputdir_glmsingle,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
    results_glmsingle['typeb'] = np.load(join(outputdir_glmsingle,'TYPEB_FITHRF.npy'),allow_pickle=True).item()
    results_glmsingle['typec'] = np.load(join(outputdir_glmsingle,'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()

elapsed_time = time.time() - start_time

print(
    '\telapsed time: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)

# %%



