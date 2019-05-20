import nibabel as nib
import numpy as np
from pathlib2 import Path
from tqdm import tqdm

def normalize(volume):
    DEFAULT_HU_MAX = 512
    DEFAULT_HU_MIN = -512
    volume = np.clip(volume, DEFAULT_HU_MIN, DEFAULT_HU_MAX)

    mxval = np.max(volume)
    mnval = np.min(volume)
    volume_norm = (volume - mnval) / max(mxval - mnval, 1e-3)

    return volume_norm


def conversion_nii2npy(root, output=None):
    root = Path(root)
    if output is None:
        output = Path(root)
    else:
        output = Path(output)

    cases = sorted([d for d in root.iterdir() if d.is_dir()])
    for case in tqdm(cases):
        vol = nib.load(str(case / 'imaging.nii.gz')).get_data()
        vol = normalize(vol)
        imaging_dir = output / case.name / 'imaging'
        if not imaging_dir.exists():
            imaging_dir.mkdir(parents=True)
        if len(list(imaging_dir.glob('*.npy'))) != vol.shape[0]:
            for i in range(vol.shape[0]):
                np.save(str(imaging_dir / f'{i:03}.npy'), vol[i])

        seg = nib.load(str(case / 'segmentation.nii.gz')).get_data()
        segmentation_dir = output / case.name / 'segmentation'
        if not segmentation_dir.exists():
            segmentation_dir.mkdir(parents=True)
        if len(list(segmentation_dir.glob('*.npy'))) != seg.shape[0]:
            for i in range(seg.shape[0]):
                np.save(str(segmentation_dir / f'{i:03}.npy'), seg[i])


if __name__ == '__main__':
    conversion_nii2npy('kits19/data', 'data')