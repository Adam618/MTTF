# -*- coding: UTF-8 -*-
import glob
import xarray
import numpy as np

# **************按时间合并nc文件*******
# path='./'
# all_files=glob.glob(path+'*.nc')
# all_files.sort()
# print(all_files[0])

# file_new=[]
# for i in range(len(all_files)):
#     file=xarray.open_dataset(all_files[i])['t2m']
#     file_new.append(file)
# da=xarray.concat(file_new,dim='time')
# da.to_netcdf('./1997to2017.nc')

# with xarray.open_dataset(path+'1995to2020.nc') as f:
#     t2m=f.t2m
# print(t2m)

# **************将温度从开尔文转为摄氏度，然后按时间拆分nc文件以npy格式*******************
import xarray as xr

# 打开原始NetCDF文件
dataset = xr.open_dataset('2010to2016.nc')
print(dataset.t2m)
dataset['t2m'] = dataset['t2m'] - 273.15

# 获取时间维度的名称
time_dim = dataset.time.dims[0]

# 逐个时间步保存为npy文件
for time_step in dataset[time_dim]:
    # 获取当前时间步的数据
    data = dataset.sel({time_dim: time_step})
    
    # 将时间转换为字符串形式
    time_str = str(time_step.dt.strftime("%Y_%m_%d_%H").values)
    
    # 去除字符串两端的引号和空格
    time_str = time_str.strip('\' ').strip('\" ')
    
    # 将数据转换为numpy数组
    data_array = data.to_array().values[:,:-1,:-1]
    data_array = np.squeeze(data_array, axis=0)
    
    # 创建新的npy文件名
    file_name = f"{time_str}.npy"
    # print(data_array.shape)
    
    # 保存为npy文件
    np.save(file_name, data_array)

# 关闭原始NetCDF文件
dataset.close()
