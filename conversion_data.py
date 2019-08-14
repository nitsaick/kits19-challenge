import multiprocessing as mp

import click
import nibabel as nib
import numpy as np
from pathlib2 import Path

from dataset import KiTS19


@click.command()
@click.option('-d', '--data', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', help='output npy file path',
              type=click.Path(dir_okay=True, resolve_path=True), required=True)
def conversion_all(data, output):
    data = Path(data)
    output = Path(output)

    cases = sorted([d for d in data.iterdir() if d.is_dir()])
    pool = mp.Pool()
    pool.map(conversion, zip(cases, [output] * len(cases)))
    pool.close()
    pool.join()


def conversion(data):
    case, output = data
    vol_nii = nib.load(str(case / 'imaging.nii.gz'))
    vol = vol_nii.get_data()
    vol = KiTS19.normalize(vol)
    
    imaging_dir = output / case.name / 'imaging'
    if not imaging_dir.exists():
        imaging_dir.mkdir(parents=True)
    if len(list(imaging_dir.glob('*.npy'))) != vol.shape[0]:
        for i in range(vol.shape[0]):
            np.save(str(imaging_dir / f'{i:03}.npy'), vol[i])

    path = case / 'segmentation.nii.gz'
    if not path.exists():
        return

    seg = nib.load(str(case / 'segmentation.nii.gz')).get_data()
    segmentation_dir = output / case.name / 'segmentation'
    if not segmentation_dir.exists():
        segmentation_dir.mkdir(parents=True)
    if len(list(segmentation_dir.glob('*.npy'))) != seg.shape[0]:
        for i in range(seg.shape[0]):
            np.save(str(segmentation_dir / f'{i:03}.npy'), seg[i])

    affine_dir = output / case.name
    if not affine_dir.exists():
        affine_dir.mkdir(parents=True)
    affine = vol_nii.affine
    np.save(str(affine_dir / 'affine.npy'), affine)


if __name__ == '__main__':
    conversion_all()
