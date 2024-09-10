# Replay-Base Architecture for Context Adaptation (RBACA)

**1. Requirements**

The requirements are given in `requirements.txt`.

**2. Data set**

Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms), https://www.ub.edu/mnms/.

After downloading it is necessary to preprocess the data by running:

```python data_prep/data_prep_cardiac.py <download_path> <dataset_path>```

where `<download_path>` is the directory where the M&Ms dataset was downloaded to, and `<dataset_path>` is where the preprocessed data is stored.

**3. Training**

To run the training, the config files in `training_configs/` are used. 

To run the base training, change to `BASE_RBACA_EVAL = 'B'` everywhere and run `python run_training.py --config training_configs/cardiac_base.yml`. Then, change the trained_model name to `cardiac_batch_cardiacdatasetsplit_base_1_d885e299cf.pt`

To run RBACA, change to `BASE_RBACA_EVAL = 'R'` everywhere and run `python run_training.py --config training_configs/cardiac_rbaca.yml`. Then, change `cardiac..._tf08_XXX_shift_Canon.pt`,`cardiac..._tf08_XXX_shift_GE.pt` and `cardiac..._tf08_XXX_shift_Philips.pt` with XXX equal to YYY in generated `cardiac_cont_cardiacdatasetsplit_basemodel_batch_memory_tf08_YYY.pt`

Finally, to perform evaluation, change to `BASE_RBACA_EVAL = 'E'` everywhere and run `python cardiac_eval.py`. To change the evaluated seeds edit `cardiac_eval.py`. The results will appear as `results.csv`.

**README.md and Code**

Adapted from Perkonigg et al. https://arxiv.org/abs/2111.13069