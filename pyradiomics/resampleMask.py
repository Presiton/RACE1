import SimpleITK as sitk
import os

image_dir = "E:/PycharmProjects/image"
mask_dir = "E:/PycharmProjects/mask"
resize_mask_dir = "E:/PycharmProjects/mask/maskresize"
img_dirs = []

for filepath, dirs, names in os.walk(image_dir):
    for name in names:
        # print(name)
        img_dir = os.path.join(image_dir, name)
        img_dirs.append(img_dir)

for img_path in img_dirs:
    file_name = img_path.split("/")[-1].split("Image.nii.gz")[0]
    print(file_name)
    mask_name = file_name + "Mask.nii.gz"
    # print(mask_name)
    mask_path = os.path.join(mask_dir,mask_name)
    # print(mask_path)
    #
    image = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(image)
    rif.SetOutputPixelType(mask.GetPixelID())
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    resMask = rif.Execute(mask)

    # True enables compression when saving the resampled mask
    sitk.WriteImage(resMask, os.path.join(resize_mask_dir + '/' + file_name + 'Resize_Mask.nii.gz'), True)


