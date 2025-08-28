import nibabel as nib
import numpy as np
import matplotlib.pylab as plt
import cvxpy as cp
import pandas as pd
from os.path import join
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
from scipy.stats import skew, kurtosis, norm
from itertools import product
from sklearn.model_selection import KFold
from matplotlib.animation import FuncAnimation
from nilearn import plotting, datasets, image


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

def plot_active_voxels(input, ses, run):
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    ax_sag, ax_cor, ax_axial = axes
    fig.suptitle(f"Less Variable Voxles-session {ses}-Run {run}")

    x = input.shape[0] // 2
    y = input.shape[1] // 2
    z = input.shape[2] // 2

    im_sag = ax_sag.imshow(np.rot90(input[x, :, :]), cmap="gray", origin='lower')
    ax_sag.set_title("Sagittal")
    ax_sag.axis("off")


    im_cor = ax_cor.imshow(np.rot90(input[:, y, :]), cmap="gray", origin='lower')
    ax_cor.set_title("Coronal")        
    ax_cor.axis("off")

    im_ax = ax_axial.imshow(np.rot90(input[:, :, z]), cmap="gray", origin='lower')
    ax_axial.set_title("Axial")   
    ax_axial.axis("off")


    def update(i):
        im_sag.set_data(np.rot90(input[i, :, :]))      # sagittal slices
        im_cor.set_data(np.rot90(input[:, i, :]))      # coronal slices
        im_ax.set_data(np.rot90(input[:, :, i]))       # axial slices
        return im_sag, im_cor, im_ax

    ani = FuncAnimation(fig, update, frames=range(min(vol.shape)), interval=100, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 1.5])
    plt.show()

    ani.save(filename=f"selected_voxels_session{ses}_run{run}.gif", writer="pillow")
    return

def plot_dist(beta_diff, run, ses):
    mu    = beta_diff.mean(axis=1)
    sigma = beta_diff.std(axis=1, ddof=1)  
    sk    = skew(beta_diff, axis=1, bias=False)
    kt    = kurtosis(beta_diff, axis=1, fisher=False, bias=False)
    rows_to_show = np.random.choice(beta_diff.shape[0], 5, replace=False)

    fig, axes = plt.subplots(len(rows_to_show), 1, figsize=(6, 2.5*len(rows_to_show)))
    for ax, idx in zip(axes, rows_to_show):
        data = beta_diff[idx]
        m, s = mu[idx], sigma[idx]
        x = np.linspace(data.min(), data.max(), 200)

        ax.hist(data, bins=25, density=True, alpha=0.5, label='data')
        ax.plot(x, norm.pdf(x, m, s), 'r', lw=2, label='Gaussian fit')
        ax.set_title(f'Row {idx}: μ={m:.2f}, σ={s:.2f}, skew={sk[idx]:.2f}, kurt={kt[idx]:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"beta_diff_selected_voxels_hist_session{ses}_run{run}.png", dpi=300, bbox_inches='tight')

    return

def plot_on_brain(anat_img, selected_img, save_path):
    display = plotting.plot_anat(anat_img, display_mode="ortho")
    display.add_overlay(selected_img, cmap="autumn", transparency=0.6, threshold=0.5)
    plotting.show()
    display.savefig(save_path)
    display.close()
    
    return

def calculate_matrices(betasmd, selected_voxels, anat_img, affine, BOLD_path_org, num_trials, trial_len):
    ## L_task Vector (contains beta values for selected voxels)##
    V1 = betasmd[selected_voxels.astype(bool), :] #(8252, 90)
    mean_V1 = np.mean(V1, axis=-1)
    L_task = 1./np.abs(mean_V1)


    BOLD_data = nib.load(BOLD_path_org).get_fdata() #(90, 128, 85, 850)
    selected_BOLD_data = BOLD_data[selected_voxels.astype(bool), :]
    selected_BOLD_data_reshape = np.zeros((selected_BOLD_data.shape[0], num_trials, trial_len))
    start = 0
    for i in range(num_trials):
        selected_BOLD_data_reshape[:, i, :] = selected_BOLD_data[:, start:start+trial_len]
        start += trial_len
        if start == 270 or start == 560:
            start += 20

    ## L_var matrix (contains variance of selected voxels)##
    diff_mat = np.diff(selected_BOLD_data_reshape, axis=1)
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
    L_smooth = csgraph.laplacian(D)

    return L_task, L_var, L_smooth, selected_BOLD_data

def objective(w, L_task, L_var, L_smooth,
              alpha_var, alpha_smooth, alpha_sparse):
    """Value of the full loss on a validation set."""
    quad = (w.T @ L_task @ w
            + alpha_var   * (w.T @ L_var    @ w)
            + alpha_smooth * (w.T @ L_smooth @ w))
    l1 = alpha_sparse * np.sum(np.abs(w))
    return quad + l1

def optimize_voxel_weights(
    L_task: np.ndarray,
    L_var: np.ndarray,
    L_smooth: np.ndarray,
    alpha_var: float = 1.0,
    alpha_smooth: float = 0.1,
    alpha_sparse: float = 0.01):
    
    L_total = np.diag(L_task) + alpha_var * L_var + alpha_smooth * L_smooth
    w = cp.Variable(L_total.shape[0])
    objective = cp.Minimize(cp.quad_form(w, L_total) + alpha_sparse * cp.norm1(w))
    problem = cp.Problem(objective)
    problem.solve(verbose=True)
    return w.value

def calculate_weight(param_grid):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_score = np.inf
    best_params = None

    for a_var, a_smooth, a_sparse in product(*param_grid.values()):
        fold_scores = []
        print(f"a_var: {a_var}, a_smooth: {a_smooth}, a_sparse: {a_sparse}")
        for train_idx, val_idx in kf.split(L_task):
            L_task_train = L_task[train_idx]
            L_var_train   = L_var[np.ix_(train_idx, train_idx)]
            L_smooth_train = L_smooth[np.ix_(train_idx, train_idx)]

            w = optimize_voxel_weights(
                L_task_train, L_var_train, L_smooth_train,
                alpha_var=a_var, alpha_smooth=a_smooth, alpha_sparse=a_sparse)

            # validation subsets
            L_task_val = L_task[val_idx]
            L_var_val   = L_var[np.ix_(val_idx, val_idx)]
            L_smooth_val = L_smooth[np.ix_(val_idx, val_idx)]

            fold_scores.append(
                objective(w, L_task_val, L_var_val, L_smooth_val,
                        a_var, a_smooth, a_sparse))

        mean_score = np.mean(fold_scores)
        print(mean_score)
        if mean_score < best_score:
            best_score = mean_score
            best_params = (a_var, a_smooth, a_sparse)

    print("Best parameters:", best_params, "with CV loss:", best_score)
    return  best_params, best_score

def select_opt_weight(selected_BOLD_data, weights, selected_voxels):
    y = selected_BOLD_data.T @ weights
    p95 = np.percentile(weights, 95)
    p5 = np.percentile(weights, 5)
    selected_weights = np.where((weights <= p5) | (weights >= p95))[0]

    weight_volume = np.zeros_like(selected_voxels, dtype=np.float32)
    weight_volume[selected_voxels.astype(bool)] = weights  # put weights in their voxel positions

    mask = np.zeros_like(weight_volume, dtype=bool)
    selected_weights = (weights <= p5) | (weights >= p95)
    mask[selected_voxels.astype(bool)] = selected_weights
    weight_volume[~mask] = 0

    masked_weights = np.where(weight_volume == 0, np.nan, weight_volume)
    weight_img = nib.Nifti1Image(masked_weights, affine=anat_img.affine)
    
    return weight_img, masked_weights, y

# %% 
t_thr = 3
R2_thr = 1.5
sk_thr = 0.1 
kt_thr = 0.2
run = 1
ses = 1
sub = '04'
num_trials = 90
trial_len = 9

param_grid = {
    "alpha_var":   [0.5, 1.0, 10.0],
    "alpha_smooth":[0.5, 0.1, 1.0],
    "alpha_sparse":[0.001, 0.01, 0.1]}

glm_result_path = '/Users/zkavian/Downloads/GLMOutputs2-sub04-ses01/TYPED_FITHRF_GLMDENOISE_RR.npy'
anat_img = nib.load('/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd004/ses-1/anat/sub-pd004_ses-1_T1w_brain_2mm.nii.gz')
base_path = '/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg_2mm.nii.gz'
BOLD_path_org = join(base_path, f'sub-pd0{sub}',f'ses-{ses}','func', data_name)

glm_results = np.load(glm_result_path, allow_pickle=True).item()
betasmd, mask_pos, mask_neg = find_active_voxels(glm_results, run, t_thr, R2_thr)  #(90,120,85,90)
active_low_var_voxels, beta_diff = find_active_low_var_voxels(betasmd, mask_pos, mask_neg, sk_thr, kt_thr)

plot_active_voxels(active_low_var_voxels, ses, run)
plot_dist(beta_diff, run, ses)

anat_data = anat_img.get_fdata()
affine = anat_img.affine
selected_voxels = nib.Nifti1Image(active_low_var_voxels.astype(np.uint8), affine)
nib.save(selected_voxels, f'affine_selected_active_low_var_voxels_session{ses}_run{run}.nii.gz')
save_path = f"anat_with_overlay(active_low_var_voxels_session{ses}_run{run}).png"
plot_on_brain(anat_img, selected_voxels, save_path)

L_task, L_var, L_smooth, selected_BOLD_data = calculate_matrices(betasmd, selected_voxels, anat_img, affine, BOLD_path_org, num_trials, trial_len)
best_params, best_score = calculate_weight(param_grid) #a_var, a_smooth, a_sparse

weights = optimize_voxel_weights(L_task, L_var, L_smooth, alpha_var=0.1, alpha_smooth=0.1, alpha_sparse=0.01)
weight_img, masked_weights, y = select_opt_weight(selected_BOLD_data, weights, selected_voxels)

save_path = f"opt_5_percent_weight_on_brain_session{ses}_run{run}.png"
plot_on_brain(anat_img, weight_img, save_path)

finish = 1
