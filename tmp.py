def find_active_voxels(glm_results, run, t_thr, R2_thr):
    if run == 1:
        betasmd = glm_results['betasmd'][:,:,:,0:90]
        R2 = glm_results['R2run'][:,:,:,0]
    else:
        betasmd = glm_results['betasmd'][:,:,:,90:]
        R2 = glm_results['R2run'][:,:,:,1]
    
    X, Y, Z, T = betasmd.shape
    V = X * Y * Z 
    B = betasmd.reshape(V, T)

    n = np.sum(~np.isnan(B), axis=1) # number of valid (non-NaN) trials for each voxel
    mu = np.nanmean(B, axis=1) 
    sd = np.nanstd(B, axis=1, ddof=1)
    se = sd / np.sqrt(np.clip(n, 1, None)); t = mu / se  #t-stat

    mu_vol = mu.reshape(X, Y, Z)
    sd_vol = sd.reshape(X, Y, Z)
    n_vol  = n.reshape(X, Y, Z)
    t_vol  = t.reshape(X, Y, Z)

    base = (R2 > R2_thr) & (np.abs(t_vol) > t_thr)
    mask_pos = base & (mu_vol > 0)
    mask_neg = base & (mu_vol < 0)

    clear_console()
    print(f"Number of Active Voxels {np.sum(base)} from Total Voxels: {V}")
    print(f"Number of Positively Active Voxels {np.sum(mask_pos)}, Number of Negatively Active Voxels {np.sum(mask_neg)}")
    
    return betasmd, mask_pos, mask_neg

def find_active_low_var_voxels(betasmd, mask_pos, mask_neg, sk_thr, kt_thr):
    mask_union = mask_pos | mask_neg
    beta_pos = betasmd[mask_union,:]
    beta_diff = np.diff(beta_pos, axis=1) #diff between consequence trial

    sk = skew(beta_diff, axis=1, bias=False)
    kt = kurtosis(beta_diff, axis=1, fisher=False, bias=False)
    mask_gaussian_like = (np.abs(sk) < sk_thr) & (np.abs(kt - 3) < kt_thr)

    selected_voxels = np.zeros(mask_pos.shape, dtype=bool)
    selected_voxels[mask_union] = mask_gaussian_like

    return selected_voxels, beta_diff

def calculate_matrices(betasmd, selected_voxels, anat_img, affine, BOLD_path_org, trial_indices, trial_len):
    ## L_task Vector (contains beta values for selected voxels)##
    num_total_trials = betasmd.shape[-1]
    if trial_indices is None:
        trial_indices = np.arange(num_total_trials)

    V1 = betasmd[selected_voxels.astype(bool), :][:, trial_indices]
    mean_V1 = np.mean(V1, axis=-1)
    L_task = np.divide(1., np.abs(mean_V1), out=np.zeros_like(mean_V1), where=mean_V1 != 0)
    # L_task = 1./np.abs(mean_V1)


    BOLD_data = nib.load(BOLD_path_org).get_fdata() #(90, 128, 85, 850)
    selected_BOLD_data = BOLD_data[selected_voxels.astype(bool), :]
    selected_BOLD_data_reshape = np.zeros((selected_BOLD_data.shape[0], num_total_trials, trial_len))
    start = 0
    for i in range(num_trials):
        selected_BOLD_data_reshape[:, i, :] = selected_BOLD_data[:, start:start+trial_len]
        start += trial_len
        if start == 270 or start == 560:
            start += 20
    selected_BOLD_data_subset = selected_BOLD_data_reshape[:, trial_indices, :]

    ## L_var matrix (contains variance of selected voxels)##
    diff_mat = np.diff(selected_BOLD_data_subset, axis=1)
    diff_mat_flat = diff_mat.reshape(diff_mat.shape[0], -1)
    L_var = np.cov(diff_mat_flat, bias=False)
    L_var = (L_var + L_var.T) / 2 + 1e-6 * np.eye(L_var.shape[0])
    # C2 = diff_mat_flat @ diff_mat_flat.T
    # L_var = C2 / selected_BOLD_data_reshape.shape[1]


    ## L_smooth matrix (contains distance beyween selected voxels)##
    anat_img_shape = anat_img.shape
    coords = np.array(np.meshgrid(
        np.arange(anat_img_shape[0]),
        np.arange(anat_img_shape[1]),
        np.arange(anat_img_shape[2]),
        indexing='ij'
    )).reshape(3, -1).T

    # Convert to world (scanner/MNI) coordinates
    world_coords = nib.affines.apply_affine(affine, coords)
    tmp = selected_voxels.astype(bool).reshape(-1)
    selected_world_coords = world_coords[tmp,:]
    D = cdist(selected_world_coords, selected_world_coords)
    sigma = np.median(D[D>0])
    W = np.exp(-D**2 / (2*sigma**2))      # similarity
    np.fill_diagonal(W, 0.0)
    L_smooth = csgraph.laplacian(W, normed=False)
    # L_smooth = csgraph.laplacian(D)

    return L_task, L_var, L_smooth, selected_BOLD_data_subset.reshape(selected_BOLD_data_subset.shape[0], -1)




t_thr = 3
R2_thr = 1.5
sk_thr = 0.1 
kt_thr = 0.2
run = 1
ses = 1
sub = '04'
num_trials = 90
trial_len = 9

# param_grid = {
#     "alpha_var":   [0.5, 1.0, 10.0],
#     "alpha_smooth":[0.5, 0.1, 1.0],
#     "alpha_sparse":[0.001, 0.01, 0.1]}

param_grid = {
    "alpha_var":   [0.5],
    "alpha_smooth":[0.5],
    "alpha_sparse":[0.001]}

glm_result_path = '/mnt/TeamShare/Data_Masterfile/Zahra-Thesis-Data/Master_Thesis_Files/GLM_single_results/GLMOutputs2-sub04-ses02/TYPED_FITHRF_GLMDENOISE_RR.npy'
anat_img = nib.load('/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd004/ses-1/anat/sub-pd004_ses-1_T1w_brain_2mm.nii.gz')
base_path = '/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg_2mm.nii.gz'
BOLD_path_org = join(base_path, f'sub-pd0{sub}',f'ses-{ses}','func', data_name)

glm_results = np.load(glm_result_path, allow_pickle=True).item()
betasmd, mask_pos, mask_neg = find_active_voxels(glm_results, run, t_thr, R2_thr)  #(90,120,85,90)
active_low_var_voxels, beta_diff = find_active_low_var_voxels(betasmd, mask_pos, mask_neg, sk_thr, kt_thr)


anat_data = anat_img.get_fdata()
affine = anat_img.affine
selected_voxels = nib.Nifti1Image(active_low_var_voxels.astype(np.uint8), affine)

kf = KFold(n_splits=2, shuffle=True, random_state=0)
best_score = np.inf
best_params = None
num_trials = betasmd.shape[-1]
alpha_var: float = 1.0
alpha_smooth: float = 0.1
alpha_sparse: float = 0.01

for a_var, a_smooth, a_sparse in product(*param_grid.values()):
    fold_scores = []
    print(f"a_var: {a_var}, a_smooth: {a_smooth}, a_sparse: {a_sparse}")
    count = 1

    for train_idx, val_idx in kf.split(np.arange(num_trials)):
        clear_console()
        print(f"k-fold num: {count}")
        L_task_train, L_var_train, L_smooth_train, _ = calculate_matrices(betasmd, active_low_var_voxels, anat_img, affine, BOLD_path_org, train_idx, trial_len)
        L_total = np.diag(L_task_train) + alpha_var * L_var_train + alpha_smooth * L_smooth_train
        n = L_total.shape[0]
        L_total = 0.5*(L_total + L_total.T) + 1e-8*np.eye(n)
        w = cp.Variable(n, nonneg=True)
        constraints = [cp.sum(w) == 1]
        
        # objective = cp.Minimize(cp.quad_form(w, L_total) + alpha_sparse * cp.norm1(w))
        objective = cp.Minimize(cp.quad_form(w, L_total))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=True)

        # w = optimize_voxel_weights(L_task_train, L_var_train, L_smooth_train, alpha_var=a_var, alpha_smooth=a_smooth, alpha_sparse=a_sparse)

        break

    break