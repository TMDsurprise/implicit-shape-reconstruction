import os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

# 指定目录路径
dir_path = Path("/home/lzl/DeepSDF/implicit-shape-reconstruction/casename_files/Colon/Task10_Colon__CTPelvic1K__fold0_3dfullres_pred")

# 遍历目录下的所有文件
for file_path in dir_path.glob('*.nii.gz'):
    # 读取volume
    img = nib.load(str(file_path))
    volume = img.get_fdata()

    # 提取值为2的voxel并计算最大连通域
    volume_2 = np.where(volume == 2, 2, 0)
    labels_2, num_features_2 = ndimage.label(volume_2)
    sizes_2 = ndimage.sum(volume_2, labels_2, range(1, num_features_2+1))
    max_label_2 = np.argmax(sizes_2) + 1
    max_connected_volume_2 = np.where(labels_2 == max_label_2, 2, 0)

    # 提取值为3的voxel并计算最大连通域
    volume_3 = np.where(volume == 3, 3, 0)
    labels_3, num_features_3 = ndimage.label(volume_3)
    sizes_3 = ndimage.sum(volume_3, labels_3, range(1, num_features_3+1))
    max_label_3 = np.argmax(sizes_3) + 1
    max_connected_volume_3 = np.where(labels_3 == max_label_3, 3, 0)

    # 保存最大连通域为新的volume
    new_img_2 = nib.Nifti1Image(max_connected_volume_2, img.affine, img.header)
    new_img_3 = nib.Nifti1Image(max_connected_volume_3, img.affine, img.header)
    nib.save(new_img_2, str(file_path.with_suffix('.pelvic_r.nii.gz')))
    nib.save(new_img_3, str(file_path.with_suffix('.pelvic_l.nii.gz')))