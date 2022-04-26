# EECS 442 Final Project (Monocular Depth Estimation)

This project is based on this [repo](https://github.com/ialhashim/DenseDepth)

## Results

### Test Image

![img](https://lh5.googleusercontent.com/4397RivXgw3_yTUXsxzI-eWoBwZwhhGUnoagD69IYtdizzdmZHR0KIfrbxUa2lcXZiSyj5z30wzYPd9I568gJf2Tl9bn1Ny9fEUPvVDQVAkggr717gh7aHBrzpqOCzQ-pHSbtlSc)



### Our Method

![img](https://lh6.googleusercontent.com/2SrtrA7aNs__P9keSwc0hAfN1PEiwBvIjGpAjZJWciOPuS2kWlZu24VbNQAgUoAl-10omu1lZOOomgmb9_HRMWcziUDTrRsxGHhAzgQttKsiNoz2W1qxYFUfZyf8Ylbq3R4Ws_oW)

## Train

Make sure to download the [NYU Depth V2 (50K)](https://tinyurl.com/nyu-data-zip)  in the same folder

To train this model

```bash
python train.py --bs=4 --epochs=4
```



## Evaluation

simply run

```
python test.py
```

and Specify which network and weights you want to specify in the *test.py* file