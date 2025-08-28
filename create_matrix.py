# %%
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt
from os.path import join
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
import cvxpy as cp
from itertools import product
from sklearn.model_selection import KFold
from matplotlib.animation import FuncAnimation
from nilearn import plotting, datasets, image
from scipy.stats import skew, kurtosis, norm
from scipy import ndimage
import pandas as pd

# %%
results = np.load('/Users/zkavian/Downloads/GLMOutputs2-sub04-ses01/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
print(results.keys())

# %%
betasmd = results['betasmd'][:,:,:,0:90]

X, Y, Z, T = betasmd.shape
V = X * Y * Z
print(f"Number of total voxels {V}")
B = betasmd.reshape(V, T)

n = np.sum(~np.isnan(B), axis=1) # number of valid (non-NaN) trials for each voxel

mu = np.nanmean(B, axis=1) # mean beta across valid trials for each voxel
sd = np.nanstd(B, axis=1, ddof=1) # std dev of beta across valid trials for each voxel
se = sd / np.sqrt(np.clip(n, 1, None)); t = mu / se  #t-stat

mu_vol = mu.reshape(X, Y, Z)
sd_vol = sd.reshape(X, Y, Z)
n_vol  = n.reshape(X, Y, Z)
t_vol  = t.reshape(X, Y, Z)

R2 = results['R2run'][:,:,:,0]

t_thr = 3
R2_thr = 1.5

base = (R2 > R2_thr) & (np.abs(t_vol) > t_thr)
print(f"Number of Active voxels {np.sum(base)}")

mask_pos = base & (mu_vol > 0)
print(f"Number of Positively Active voxels {np.sum(mask_pos)}")
mask_neg = base & (mu_vol < 0)
print(f"Number of Negatively Active voxels {np.sum(mask_neg)}")

# %%
betasmd = results['betasmd']
mask_union = mask_pos | mask_neg
beta_pos = betasmd[mask_union,:]
# print(beta_pos.shape) #(87656, 180)

beta_pos1 = beta_pos[:,:90]
beta_pos2 = beta_pos[:,90:]

beta_diff1 = np.diff(beta_pos1, axis=1)
beta_diff2 = np.diff(beta_pos2, axis=1)

sk1 = skew(beta_diff1, axis=1, bias=False)
kt1 = kurtosis(beta_diff1, axis=1, fisher=False, bias=False)
sk2 = skew(beta_diff2, axis=1, bias=False)
kt2 = kurtosis(beta_diff2, axis=1, fisher=False, bias=False)

sk_thr = 0.1                        # |sk| < 0.1 → near-symmetric
kt_thr = 0.2                        # |kt-3| < 0.2 → near-Normal kurtosis

# voxels meeting both criteria
mask_gaussian_like = (np.abs(sk1) < sk_thr) & (np.abs(kt1 - 3) < kt_thr)
selected_voxels_run1 = np.zeros(mask_pos.shape, dtype=bool)
selected_voxels_run1[mask_union] = mask_gaussian_like
print(selected_voxels_run1.shape)

mask_gaussian_like = (np.abs(sk2) < sk_thr) & (np.abs(kt2 - 3) < kt_thr)
selected_voxels_run2 = np.zeros(mask_pos.shape, dtype=bool)
selected_voxels_run2[mask_union] = mask_gaussian_like
print(selected_voxels_run2.shape)

# %%
def plot_active_voxels_sech(vol, ses, run):
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    ax_sag, ax_cor, ax_axial = axes
    fig.suptitle(f"Less Variable Voxles-session {ses}-Run {run}")

    x = vol.shape[0] // 2
    y = vol.shape[1] // 2
    z = vol.shape[2] // 2

    im_sag = ax_sag.imshow(np.rot90(vol[x, :, :]), cmap="gray", origin='lower')
    ax_sag.set_title("Sagittal")
    ax_sag.axis("off")


    im_cor = ax_cor.imshow(np.rot90(vol[:, y, :]), cmap="gray", origin='lower')
    ax_cor.set_title("Coronal")        
    ax_cor.axis("off")

    im_ax = ax_axial.imshow(np.rot90(vol[:, :, z]), cmap="gray", origin='lower')
    ax_axial.set_title("Axial")   
    ax_axial.axis("off")


    def update(i):
        im_sag.set_data(np.rot90(vol[i, :, :]))      # sagittal slices
        im_cor.set_data(np.rot90(vol[:, i, :]))      # coronal slices
        im_ax.set_data(np.rot90(vol[:, :, i]))       # axial slices
        return im_sag, im_cor, im_ax

    ani = FuncAnimation(fig, update, frames=range(min(vol.shape)), interval=100, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 1.5])
    plt.show()

    ani.save(filename=f"selected_voxels_session{ses}_run{run}.gif", writer="pillow")
    return

# %%
ses = 1
run = 1
plot_active_voxels_sech(selected_voxels_run1, ses, run)

run = 2
plot_active_voxels_sech(selected_voxels_run2, ses, run)

# %%
def plot_dist(beta_diff):
    mu    = beta_diff.mean(axis=1)
    sigma = beta_diff.std(axis=1, ddof=1)          # sample std
    sk    = skew(beta_diff, axis=1, bias=False)
    kt    = kurtosis(beta_diff, axis=1, fisher=False, bias=False)

    # Plot a few example rows
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

    return

# %%
plot_dist(beta_diff1)

# %%
anat_img = nib.load('/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd004/ses-1/anat/sub-pd004_ses-1_T1w_brain_2mm.nii.gz')
anat_data = anat_img.get_fdata()
affine = anat_img.affine
ses = 1
run = 1

selected_img = nib.Nifti1Image(selected_voxels_run1.astype(np.uint8), affine)
nib.save(selected_img, f'selected_voxels_session{ses}_run{run}.nii.gz')
display = plotting.plot_anat(anat_img, display_mode="ortho")
display.add_overlay(selected_img, cmap="autumn", transparency=0.6, threshold=0.5)
plotting.show()

run = 2
selected_img = nib.Nifti1Image(selected_voxels_run2.astype(np.uint8), affine)
nib.save(selected_img, f'selected_voxels_session{ses}_run{run}.nii.gz')
display = plotting.plot_anat(anat_img, display_mode="ortho")
display.add_overlay(selected_img, cmap="autumn", transparency=0.6, threshold=0.5)
plotting.show()

# %%
def determine_rois(data_path, ses, run):
    stat_img = nib.load(data_path)
    stat = stat_img.get_fdata()
    affine = stat_img.affine

    # Treat anything >0 as part of a cluster (adapt if you want a value threshold)
    mask = stat > 0

    # Use 26-connectivity for 3D clusters
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labels, n_clusters = ndimage.label(mask, structure=structure)

    if n_clusters == 0:
        raise SystemExit("No nonzero voxels found in the image.")

    # ---------- 2) Find cluster props: size, peak voxel, center of mass ----------
    rows = []
    for c in range(1, n_clusters + 1):
        # indices in ijk (voxel) space
        vox_idx = np.argwhere(labels == c)
        values = stat[labels == c]
        n_vox = vox_idx.shape[0]

        # peak voxel (by value; if your map is binary, it's an arbitrary voxel in the cluster)
        peak_i = np.argmax(values)
        peak_ijk = vox_idx[peak_i]

        # center of mass in voxel coords (floating point)
        com_ijk = ndimage.center_of_mass(mask, labels, c)

        # convert to MNI (RAS) coordinates
        peak_xyz = nib.affines.apply_affine(affine, peak_ijk)
        com_xyz = nib.affines.apply_affine(affine, com_ijk)

        rows.append({
            "cluster_id": c,
            "n_voxels": int(n_vox),
            "peak_value": float(values[peak_i]),
            "peak_i": int(peak_ijk[0]), "peak_j": int(peak_ijk[1]), "peak_k": int(peak_ijk[2]),
            "peak_x": float(peak_xyz[0]), "peak_y": float(peak_xyz[1]), "peak_z": float(peak_xyz[2]),
            "com_x": float(com_xyz[0]), "com_y": float(com_xyz[1]), "com_z": float(com_xyz[2]),
        })

    df = pd.DataFrame(rows)

    # ---------- 3) Fetch Harvard–Oxford atlases & resample to your image ----------
    # Max-probability (discrete labels) at 2mm, thr=25%
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_sub  = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    cort_img = image.load_img(ho_cort.maps)
    sub_img  = image.load_img(ho_sub.maps)

    # Resample to the same grid as your stat image so we can index by ijk directly
    cort_res = image.resample_to_img(cort_img, stat_img, interpolation='nearest')
    sub_res  = image.resample_to_img(sub_img,  stat_img, interpolation='nearest')

    cort_arr = cort_res.get_fdata().astype(int)
    sub_arr  = sub_res.get_fdata().astype(int)

    # Make label lookup dicts (index -> region name)
    cort_labels = {i: name for i, name in enumerate(ho_cort.labels)}
    sub_labels  = {i: name for i, name in enumerate(ho_sub.labels)}

    def name_or_empty(label_idx, lut):
        return lut.get(int(label_idx), "") if int(label_idx) != 0 else ""

    # ---------- 4) Assign a label to each cluster (at the PEAK voxel) ----------
    cort_names = []
    sub_names  = []

    for _, r in df.iterrows():
        i, j, k = int(r.peak_i), int(r.peak_j), int(r.peak_k)
        # bounds check (rarely needed if resampling matched shapes)
        if not (0 <= i < cort_arr.shape[0] and 0 <= j < cort_arr.shape[1] and 0 <= k < cort_arr.shape[2]):
            cort_names.append("")
            sub_names.append("")
            continue
        cort_lab = cort_arr[i, j, k]
        sub_lab  = sub_arr[i, j, k]
        cort_names.append(name_or_empty(cort_lab, cort_labels))
        sub_names.append(name_or_empty(sub_lab,  sub_labels))

    df["HO_cortical_label"]    = cort_names
    df["HO_subcortical_label"] = sub_names

    # Optional: a single best label field preferring cortical, then subcortical
    def best_label(cort, sub):
        return cort if cort else sub
    df["best_label"] = [best_label(c, s) for c, s in zip(df["HO_cortical_label"], df["HO_subcortical_label"])]

    # ---------- 5) Save results ----------
    out_csv = f"selected_voxels_HO_labels-ses{ses}-run{run}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nFound {len(df)} clusters. Saved: {out_csv}")
    print(df[["cluster_id","n_voxels","peak_value","peak_x","peak_y","peak_z","best_label"]])

    return

# %%
determine_rois("/Users/zkavian/Desktop/Workspace/Python_code/GLMSingle/selected_voxels_session1_run1.nii.gz", 1, 1)
determine_rois("/Users/zkavian/Desktop/Workspace/Python_code/GLMSingle/selected_voxels_session1_run2.nii.gz", 1, 2)


# %% [markdown]
# # Run1

# %%
data_path = '/Users/zkavian/Desktop/Workspace/Python_code/GLMSingle/selected_voxels_session1_run1.nii.gz'
stat_img = nib.load(data_path)
selected_voxels = stat_img.get_fdata() #binary matrix (90,120,85)

results = np.load('/Users/zkavian/Downloads/GLMOutputs2-sub04-ses01/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_values = betasmd = results['betasmd'][:,:,:,0:90]  #(90,120,85,90)

v1 = beta_values[selected_voxels.astype(bool), :] #(8252, 90)
mean_v1 = np.mean(v1, axis=-1)
L_task = 1./np.abs(mean_v1)


# %%
base_path = '/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
sub = '04'
ses = '2'
run = '1'

data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg_2mm.nii.gz'
data_path_org = join(base_path, f'sub-pd0{sub}',f'ses-{ses}','func', data_name)
data_run1 = nib.load(data_path_org).get_fdata() #(90, 128, 85, 850)

# %%
select_data = data_run1[selected_voxels.astype(bool), :]

num_trials = 90
trial_len = 9
active_voxel_data = np.zeros((select_data.shape[0], num_trials, trial_len))

start = 0
for i in range(num_trials):
    active_voxel_data[:, i, :] = select_data[:, start:start+trial_len]
    start += trial_len
    if start == 270 or start == 560:
        start += 20


# %%
diff_mat = np.diff(active_voxel_data, axis=1)
diff_mat_flat = diff_mat.reshape(diff_mat.shape[0], -1)
C2 = diff_mat_flat @ diff_mat_flat.T
L_var = C2 / active_voxel_data.shape[1]
# C2.shape

# %%
anat_img = nib.load('/Volumes/McKeownLab/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd004/ses-1/anat/sub-pd004_ses-1_T1w_brain_2mm.nii.gz')
anat_data = anat_img.get_fdata()

# %%
affine = anat_img.affine
shape = anat_img.shape

# Get all voxel indices
coords = np.array(np.meshgrid(
    np.arange(shape[0]),
    np.arange(shape[1]),
    np.arange(shape[2]),
    indexing='ij'
)).reshape(3, -1).T

# Convert to world (scanner/MNI) coordinates
world_coords = nib.affines.apply_affine(affine, coords)
tmp = selected_voxels.astype(bool).reshape(-1)
selected_world_coords = world_coords[tmp,:]

D = cdist(selected_world_coords, selected_world_coords)  

L_smooth = csgraph.laplacian(D)



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
    problem.solve()
    return w.value


param_grid = {
    "alpha_var":   [0.1, 1.0, 10.0],
    "alpha_smooth":[0.0, 0.1, 1.0],
    "alpha_sparse":[0.001, 0.01, 0.1],
}

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

weights = optimize_voxel_weights(
    L_task, L_var, L_smooth, alpha_var=0.1, alpha_smooth=0.1, alpha_sparse=0.01
)
# print("Optimized weights shape:", weights.shape)
y = select_data.T @ weights

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

display = plotting.plot_anat(anat_img, dim=-0.5, title="Selected weights")
display.add_overlay(weight_img, cmap="cold_hot", threshold=1e-6)
plotting.show()



