# KPFBNC

## Requirement
- torch                   2.1.2+cu118
- torch_geometric         2.5.3
- numpy                   1.23.0
- pandas                  2.0.3
- scikit-learn            1.3.2
- matplotlib              3.7.3
- seaborn                 0.13.0
- nilearn                 0.10.2

## Datasets
### ABIDE
The ABIDE dataset is open access to all users, we have provide the time series data from ABIDE dataset in ./TimeSeries directory, 
or you can download it [follow the instructions on](http://preprocessed-connectomes-project.org/abide/download.html)

### ADHD
The ADHD dataset could be [downloaded from](http://preprocessed-connectomes-project.org/adhd200/download.html)

### HCP
The HCP dataset could be [downloaded from](https://db.humanconnectome.org/)

## Command
python main.py --data_name 'ABIDE200'


