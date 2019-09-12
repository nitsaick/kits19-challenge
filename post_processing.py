import cc3d
import click
import nibabel as nib
from pathlib2 import Path
from tqdm import tqdm


@click.command()
@click.option('-d', '--data', help='prediction data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', help='output path',
              type=click.Path(dir_okay=True, resolve_path=True), required=True)
def main(data, output):
    data = Path(data).resolve()
    output = Path(output).resolve()
    
    assert data != output, f'postprocess data will replace original data, use another output path'
    
    if not output.exists():
        output.mkdir(parents=True)
    
    predictions = sorted(data.glob('prediction_*.nii.gz'))
    for pred in tqdm(predictions):
        vol_nii = nib.load(str(pred))
        affine = vol_nii.affine
        vol = vol_nii.get_data()
        vol = post_processing(vol)
        vol_nii = nib.Nifti1Image(vol, affine)
        
        vol_nii_filename = output / pred.name
        vol_nii.to_filename(str(vol_nii_filename))


def post_processing(vol):
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background
    reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.1]
    for i in reduce_cc:
        vol[vol_cc == i] = 0
    
    return vol


if __name__ == '__main__':
    main()
