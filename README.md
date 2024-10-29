## High-accuracy prediction of 5-year progression-free melanoma survival using spatial proteomics and few-shot learning on primary tissue
1) We provide the following data to reproduce the results from different starting points. To repeat the analyses and create the plots from our paper, you can: 
- clone this github repository
- download the image data, the metadata, the anndata file, and the model weights from zenodo
- create an anaconda environment based on the provided environment.yml file
- run the jupyter notebooks to re-produce the figures

2) If you want to train the models yourself (instead of using our provided weights), we provide the resized (512x512) image data and the required python scripts:
    1) Download these publicly available weights https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent
    2) Model pre-training
    - pre-train and specify section IDs of validation samples, or leave out --val_samples argument for pre-training on all data
    - in pre-training, the validation set needs to contain samples from both classes!
    - we used the index parameter to differentiate between the cross-validation runs
        ```
        python model/melanoma_main.py --config_path ../config.json --device "cuda:0" --pretraining --index 0 --val_samples 0 1 2
        ```
        - to pre-train with specific samples left out:
        ```
        python model/melanoma_main.py --config_path ../config.json --device "cuda:0" --pretraining  --index 0 --leave_out 0 
        ```
    3) Model fine-tuning
    - specifiy validation samples for specific run:
    ```
    python model/melanoma_main.py --config_path ../config.json --device "cuda:0" --finetuning --index 0 --val_samples 0 1 2 
    ```
    - here, the validation set is of one class

3) If you want to comprehend how we performed segmentation and marker expression analysis, you can run these, too. Please be aware that the results will differ from the results in the provided anndata files, as we ran the segmentation and expression analysis on the original data (2018x2018 pixels), which we cannot upload due to the enourmous size. 
    - from https://www.proteinatlas.org/about/download download the Human Protein Atlas reference data (files rna_single_cell_type_tissue.tsv.zip and
rna_single_cell_cluster_description.tsv.zip), unzip them, and add their path to the config file
    - navigate to src
    - run 
    ```
    python segmentation/create_melc_anndata_melanoma.py --config_path=../config.json
    ```

4) If you want to comprehend how we retrieved the regions of interest, use the jupyter notebook generate_smoothgrad_maps.ipynb
