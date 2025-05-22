# RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer

RESTORE-DiT is **a novel Diffusion-based framework** for Satellite Image Time Series (SITS) reconstruction. Conditioned on **SAR image time series** and **date information**, RESTORE-DiT achieves superior reconstruction performance for highly-dynamic land surface (e.g. vegetations) with persist cloud cover.

![Figure 6](https://github.com/user-attachments/assets/7a4e4363-8f6b-44e2-b8f7-0e8f129d4736)



## :speech_balloon: To do list
- [x] Code and configuration for training and test dataset at France site.
- [x] Code for Denoising Transformer.
- [ ] Training code of RESTORE-DiT.
- [ ] Inference code of RESTORE-DiT.



## :speech_balloon: Data preparation

1. **Dataset download**

    Download the original PASTIS-R dataset [here](https://zenodo.org/records/5735646).

2. **Generate Cloud masks**

   The model is trained on cloud-free image time series with simulated masks. Cloud/shadow detection is necessary to remove cloudy images in each time series. [CloudSEN12](https://github.com/cloudsen12) is used for cloud/shadow detection.

## :speech_balloon: Training

    Coming soon...

## :speech_balloon: Evaluation

    Coming soon...
    
## Acknowledgements
