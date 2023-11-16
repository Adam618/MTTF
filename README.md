# MTTF: A Multimodal Transformer for Temperature
Forecasting
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the source code for our paper: A Multimodal Transformer for Temperature

<p align="center">
<img src=".\img\model.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of MTTF.
</p>

## Abstract

Accurate weather forecasting is important for various applications, including agriculture and environmental monitoring. However, previous methods typically use only temperature observations as input, which does not take into account spatial location(e.g., neighboring regions usually show similar temperature trends) and makes it challenging to predict abrupt changes in temperature caused by weather effects. To address this issue, we propose a multimodal Transformer for temperature forecasting that utilizes both weather station temperature observations (time series data) and ERA5 temperature reanalysis data (spatio-temporal data) as input. Our method has several features, including the use of multiple sources of data for more accurate predictions, a cross-all-modal attention mechanism to capture dependencies between modalities, and a sequence merging module to reduce the number of attention calculations and improve prediction accuracy. We evaluate our method on two real temperature datasets and show that it outperforms existing methods in several metrics, offering a promising new solution for temperature prediction.



<p align="center">
<img src=".\img\visual_new.bmp" height = "400" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of MTTF
</p>

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data 
The experiment utilized temperature observations and their corresponding reanalysis data from three regions: Delhi(India), Shiquanhe(China), Cuona(China). 
### Temperature observations
The observations for Delhi can be obtained for free on Kaggle and belongs to Weather Underground (can be downloaded [here](https://www.kaggle.com/datasets/mahirkukreja/delhi-weather-data)).

The observation data for Cuona and Shiquanhe were obtained from the National Data Center of the China Meteorological Administration (can be downloaded [here](https://data.cma.cn/dataService/cdcindex/datacode/A.0012.0001/show_value/normal.html)).

We have downloaded and processed the observations (filled in missing values, etc.) and placed them in the appropriate folder `data/region_name`. The observations processing script can be found at `data/region_name/csv_script.py`.


### Temperature reanalysis data
Temperature reanalysis data for the three regions can be downloaded free of charge from the European Center for Medium-Range Weather Forecasts (ECMWF) website (can be downloaded [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)).





## Usage

```bash
# Downloading and Processing Reanalysis Data for Shiquanhe
python -u data/shiquanhe/nc_script.py

```


```bash
# Shiquanhe
python -u main_informer.py --model informer --data custom --learning_rate 0.0001 --batch_size 32 --seq_len 2 --label_len 2 --pred_len 8 --img_pred_len 0 --features S  --d_model 512 --img_channel 2  --device 0 --lag_step 0 --patience 10  --img_path t_q_500pa   --itr 2 --attn full  --is_img_embed --e_layers 3 --data_path station_20010102_22043023.csv --distil


```
The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment.  |
| data           | The dataset name                                             |
| root_path      | The root path of the data file     |
| data_path      | The data file name                 |
| features       | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate) |
| target         | Target feature in S or MS task (defaults to `OT`)             |
| freq           | Freq for time features encoding (defaults to `h`). This can be set to `s`,`t`,`h`,`d`,`b`,`w`,`m` (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h |
| checkpoints    | Location of model checkpoints (defaults to `./checkpoints/`)  |
| seq_len | Input sequence length of Informer encoder (defaults to 96) |
| label_len | Start token length of Informer decoder (defaults to 48) |
| pred_len | Prediction sequence length (defaults to 24) |
| enc_in | Encoder input size (defaults to 7) |
| dec_in | Decoder input size (defaults to 7) |
| c_out | Output size (defaults to 7) |
| d_model | Dimension of model (defaults to 512) |
| n_heads | Num of heads (defaults to 8) |
| e_layers | Num of encoder layers (defaults to 2) |
| d_layers | Num of decoder layers (defaults to 1) |
| s_layers | Num of stack encoder layers (defaults to `3,2,1`) |
| d_ff | Dimension of fcn (defaults to 2048) |
| padding | Padding type(defaults to 0). |
| distil | Whether to use distilling in encoder, using this argument means not using distilling (defaults to `True`) |
| dropout | The probability of dropout (defaults to 0.05) |
| embed | Time features encoding (defaults to `timeF`). This can be set to `timeF`, `fixed`, `learned` |
| activation | Activation function (defaults to `gelu`) |
| output_attention | Whether to output attention in encoder, using this argument means outputing attention (defaults to `False`) |
| do_predict | Whether to predict unseen future data, using this argument means making predictions (defaults to `False`) |
| mix | Whether to use mix attention in generative decoder, using this argument means not using mix attention (defaults to `True`) |
| cols | Certain cols from the data files as the input features |
| num_workers | The num_works of Data loader (defaults to 0) |
| itr | Experiments times (defaults to 2) |
| train_epochs | Train epochs (defaults to 6) |
| batch_size | The batch size of training input data (defaults to 32) |
| patience | Early stopping patience (defaults to 3) |
| learning_rate | Optimizer learning rate (defaults to 0.0001) |
| des | Experiment description (defaults to `test`) |
| loss | Loss function (defaults to `mse`) |
| lradj | Ways to adjust the learning rate (defaults to `type1`) |
| use_amp | Whether to use automatic mixed precision training, using this argument means using amp (defaults to `False`) |
| inverse | Whether to inverse output data, using this argument means inversing output data (defaults to `False`) |
| use_gpu | Whether to use gpu (defaults to `True`) |
| gpu | The gpu no, used for training and inference (defaults to 0) |
| use_multi_gpu | Whether to use multiple gpus, using this argument means using mulitple gpus (defaults to `False`) |
| devices | Device ids of multile gpus (defaults to `0,1,2,3`) |


More parameter information please refer to `main_MTTF.py`.



#

## Contact
If you have any questions, feel free to contact Yang Cao through Email (cy5115236@163.com.com) or Github issues. 


