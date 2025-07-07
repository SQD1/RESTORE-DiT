# [RSE 2025] RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer

RESTORE-DiT is **a novel Diffusion-based framework** for Satellite Image Time Series (SITS) reconstruction. Our work **firstly** promotes the sequence-level optical-SAR fusion through a **diffusion** framework. [Paper](https://www.sciencedirect.com/science/article/pii/S0034425725002767).

Inspired by the great success of diffusion models in image and video generation, we approach the time series reconstruction problem from the perspective of **conditional generation**. Conditioned on **SAR image time series** and **date information**, RESTORE-DiT achieves superior reconstruction performance for **highly-dynamic** land surface (e.g. vegetations) under **persist cloud cover** (as shown below).

![Figure 6](https://github.com/user-attachments/assets/7a4e4363-8f6b-44e2-b8f7-0e8f129d4736)


## :speech_balloon: Method overview

![Figure 4](https://github.com/user-attachments/assets/bec7e831-037b-49ac-9c5d-702bdd5bf229)
Fig. 1. Structure of RESTORE-DiT framework. The noisy cloudy optical time series is iteratively denoised by Denoising Transformer under the condition of SAR and date.


## :speech_balloon: To do list
- [x] Code and configuration for training and test dataset at France site.
- [x] Code for Denoising Transformer.
- [x] Training codes of RESTORE-DiT.
- [x] Evaluation codes of RESTORE-DiT.

## :speech_balloon: Requirments
```bash
pip install -r requirements.txt
```




## :speech_balloon: Data preparation

1. **Dataset download**

    Download the original PASTIS-R dataset [here](https://zenodo.org/records/5735646).

2. **Generate cloud masks**

   The model is trained on cloud-free image time series with simulated masks. Cloud/shadow detection is necessary to remove cloudy images in each time series. [CloudSEN12](https://github.com/cloudsen12) is used for cloud/shadow detection.

   Modify the data folder of PASTIS-R in `CloudDetection.py` and generate the real cloud masks of PASTIS-R dataset. You may need to install necessary packages like segmentation_models_pytorch and geopandas to run `CloudDetection.py`.

3. **Record cloudy frames**

   Based on the obtained cloud masks [T,1,H,W], Indexes of cloudy frames for each sample are recorded in a json file, which will be used for pre-processing in [PASTISDataset.py](https://github.com/SQD1/RESTORE-DiT/blob/main/lib/datasets/PASTISDataset.py). We provide the [json](https://github.com/SQD1/RESTORE-DiT/blob/main/lib/datasets/bad_frames.json) file for the PASTIS-R dataset. You can simply place it to the root of PASTIS-R dataset for training.

## :speech_balloon: Training

    python run_train_PASTIS.py ./configs/config_PASTIS_train.yaml --save_dir ./results/

    
This command will create a `./results/START_TIME` path, which saves the training configs and models. The START_TIME is the folder named based on the time you start training, which could be shown as "2025-06-17_18-00".

## :speech_balloon: Evaluation

    python run_eval.py config_yaml_path SDT --test-data.test-config ./configs/config_PASTIS_test_simulation.yaml --checkpoint pth_model_path --inference_steps 1

To use the command above for evaluation on the test set of PASTIS-R, YOU NEED TO:
1. Replace the `config_yaml_path` to your specific config.yaml path in `results` folder, which could be like `./results/2025-06-17_18-00/config.yaml`. 
2. Replace the `pth_model_path` to your specific saved model path in `results` folder, which could be like `./results/2025-06-17_18-00/checkpoints/Model_best.pth`.


## :speech_balloon: Citation 

If you find our method useful in your research, please cite with:

```
@ARTICLE{RESTORE-DiT,
  author={Shu, Qidi and Zhu, Xiaolin and Xu, Shuai and Wang, Yan and Liu, Denghong},
  journal={Remote Sensing of Environment}, 
  title={RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer}, 
  year={2025},
  volume={328},
  number={114872},
}
```


## Acknowledgements

Thanks for these excellent works: [U-TILISE](https://github.com/prs-eth/U-TILISE), [VDT](https://github.com/RERV/VDT), [DiT](https://github.com/facebookresearch/DiT), [DiffCR](https://github.com/XavierJiezou/DiffCR), [PASTIS-R](https://github.com/VSainteuf/pastis-benchmark).

