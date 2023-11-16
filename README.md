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
Temperature reanalysis data for the three regions can be downloaded free of charge from the European Center for Medium-Range Weather Forecasts (ECMWF) website (can be downloaded [here](https://data.cma.cn/dataService/cdcindex/datacode/A.0012.0001/show_value/normal.html)).





## Usage


```bash
# Shiquanhe
python -u main_informer.py --model informer --data custom --learning_rate 0.0001 --batch_size 32 --seq_len 2 --label_len 2 --pred_len 8 --img_pred_len 0 --features S  --d_model 512 --img_channel 2  --device 0 --lag_step 0 --patience 10  --img_path t_q_500pa   --itr 2 --attn full  --is_img_embed --e_layers 3 --data_path station_20010102_22043023.csv --distil


```

More parameter information please refer to `main_MTTF.py`.



#

## Contact
If you have any questions, feel free to contact Yang Cao through Email (cy5115236@163.com.com) or Github issues. 


