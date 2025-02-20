# KPFBNC

## Datasets
### ABIDE
The ABIDE dataset is openly accessible to all users. We have provided the time series data from the ABIDE dataset in the ./TimeSeries directory. Alternatively, you can download it by following the instructions on the [ABIDE website](http://preprocessed-connectomes-project.org/abide/download.html).

### ADHD
The ADHD dataset can be downloaded from the [ADHD website](http://preprocessed-connectomes-project.org/adhd200/download.html).

### HCP
The HCP dataset can be downloaded from the [HCP website](https://db.humanconnectome.org/).

## Command
python main.py --data_name 'ABIDE200'

## Requirement
- torch                   2.1.2+cu118
- torch_geometric         2.5.3
- numpy                   1.23.0
- pandas                  2.0.3
- scikit-learn            1.3.2
- matplotlib              3.7.3
- seaborn                 0.13.0
- nilearn                 0.10.2
