import os
import numpy as np
import segmentation_models_pytorch as smp
import xarray as xr
from tqdm import tqdm
import torch
from torch.utils.model_zoo import load_url
import geopandas as gpd

# borrowed from https://github.com/earthnet2021/earthnet-minicuber/blob/2e29d214f279b24aad73244aa7c415a82ee6c69f/earthnet_minicuber/provider/s2/cloudmask.py

def get_checkpoint(bands_avail):
    bands_avail = set(bands_avail)

    if set(['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'WVP']).issubset(
            bands_avail):
        ckpt = load_url("https://nextcloud.bgc-jena.mpg.de/s/qHKcyZpzHtXnzL2/download/mobilenetv2_l2a_all.pth")
        ckpt_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'WVP']
    elif set(["B02", "B03", "B04", "B8A"]).issubset(bands_avail):
        ckpt = load_url("https://nextcloud.bgc-jena.mpg.de/s/Ti4aYdHe2m3jBHy/download/mobilenetv2_l2a_rgbnir.pth")
        ckpt_bands = ["B02", "B03", "B04", "B8A"]
    else:
        raise Exception(
            f"The bands {bands_avail} do not contain the necessary bands for cloud masking. Please include at least bands B02, B03, B04 and B8A.")
        ckpt = None
        ckpt_bands = None

    return ckpt, ckpt_bands


class CloudMask:
    def __init__(self, bands=["B02", "B03", "B04", "B8A"], cloud_mask_rescale_factor=None):

        self.cloud_mask_rescale_factor = cloud_mask_rescale_factor
        self.bands = bands
        ckpt, self.ckpt_bands = get_checkpoint(bands)

        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            classes=4,
            in_channels=len(self.ckpt_bands)
        )

        if ckpt:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        self.bands_scale = xr.DataArray(12 * [10000, ] + [65535, 65535, 1], coords={
            "band": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AOT", "WVP",
                     "SCL"]})

    def __call__(self, stack):

        x = torch.from_numpy((stack / 10000.0).astype("float32"))

        b, c, h, w = x.shape

        h_big = ((h // 32 + 1) * 32)
        h_pad_left = (h_big - h) // 2
        h_pad_right = ((h_big - h) + 1) // 2

        w_big = ((w // 32 + 1) * 32)
        w_pad_left = (w_big - w) // 2
        w_pad_right = ((w_big - w) + 1) // 2

        x = torch.nn.functional.pad(x, (w_pad_left, w_pad_right, h_pad_left, h_pad_right), mode="reflect")

        if self.cloud_mask_rescale_factor:
            # orig_size = (x.shape[-2], x.shape[-1])
            x = torch.nn.functional.interpolate(x, scale_factor=self.cloud_mask_rescale_factor, mode='bilinear')

        with torch.no_grad():
            y_hat = self.model(x)

        y_hat = torch.argmax(y_hat, dim=1).float()

        if self.cloud_mask_rescale_factor:
            y_hat = torch.nn.functional.max_pool2d(y_hat[:, None, ...], kernel_size=self.cloud_mask_rescale_factor)[:,
                    0, ...]  # torch.nn.functional.interpolate(y_hat, size = orig_size, mode = "bilinear")

        y_hat = y_hat[:, h_pad_left:-h_pad_right, w_pad_left:-w_pad_right]
        y_hat = y_hat.unsqueeze(1)

        mask = y_hat.cpu().numpy()
        return mask


def cloud_mask_reduce(x, axis=None, **kwargs):
    return np.where((x == 1).any(axis=axis), 1, np.where((x == 3).any(axis=axis), 3,
                                                         np.where((x == 2).any(axis=axis), 2,
                                                                  np.where((x == 0).any(axis=axis), 0, 4))))

def Generate_cloud_mask(model, id_patches, save_root):

    for id in tqdm(id_patches):
        # read a sequence
        sequence = np.load(os.path.join(folder, "DATA_S2", "S2_{}.npy".format(id)))
        # sequence = np.load(os.path.join(folder, "DATA_S2", id))

        # cloud detection
        # bands = [1, 2, 3, 8]   # [B02, B03, B04, B08A]
        bands = [0, 1, 2, 7]     # [B02, B03, B04, B08A]
        sequence = sequence[:, bands, :, :]
        mask = model(sequence)
        mask = mask.astype(np.int8)    # [T, 1, H, W]
        np.save(os.path.join(save_root, str(id)+'.npy'), mask)


if __name__ == "__main__":
    """
        Generate real cloud masks of PASTIS-R dataset.
        Category in mask:
            0: clear
            1: Thick cloud
            2: Thin cloud
            3. Cloud shadow
    """

    # Modify the following input path and save path
    # The Cloud masks are saved in the REAL_MASKS folder of PASTIS-R path
    folder = r'/data0/qidi/PASTIS-R'
    save_root = r'/data0/qidi/PASTIS-R/REAL_MASKS'

    os.makedirs(save_root, exist_ok=True)

    model = CloudMask(bands=["B02", "B03", "B04", "B8A"], cloud_mask_rescale_factor=None)

    # id_patches
    meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
    meta_patch.index = meta_patch["ID_PATCH"].astype(int)
    meta_patch.sort_index(inplace=True)
    id_patches = meta_patch.index

    # id_patches = os.listdir(os.path.join(folder, "DATA_S2"))

    Generate_cloud_mask(model, id_patches, save_root)
