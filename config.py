from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for dataset paths and identifiers."""
    subject: str = "04"
    session: int = 1
    run: int = 1
    base_path: Path = Path(
        "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
        "PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives"
    )
    glm_dir: Path = Path(
        "/mnt/TeamShare/Data_Masterfile/Zahra-Thesis-Data/"
        "Master_Thesis_Files/GLM_single_results"
    )

    @property
    def glm_result_path(self) -> Path:
        return self.glm_dir / f"GLMOutputs2-sub{self.subject}-ses0{self.session}/TYPED_FITHRF_GLMDENOISE_RR.npy"

    @property
    def anat_path(self) -> Path:
        return (
            self.base_path
            / f"sub-pd0{self.subject}"
            / f"ses-{self.session}"
            / "anat"
            / f"sub-pd0{self.subject}_ses-{self.session}_T1w_brain_2mm.nii.gz"
        )

    @property
    def bold_path(self) -> Path:
        data_name = (
            f"sub-pd0{self.subject}_ses-{self.session}_run-{self.run}_"
            "task-mv_bold_corrected_smoothed_reg_2mm.nii.gz"
        )
        return (
            self.base_path
            / f"sub-pd0{self.subject}"
            / f"ses-{self.session}"
            / "func"
            / data_name
        )