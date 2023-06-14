# Quantization Results

## PerChannelMinMaxQuantization for weights

Results:

| Model           | Accuracy | Loss | Time Performing |
|-----------------|----------|------|-----------------|
| Before Quantize | 70.98    | 1.18 | 39.43           |
| After Quantize  | 58.48    | 1.75 | 9.81            |

## MinMaxQuantization (PerTensor) for weights

Results:

| Model           | Accuracy | Loss | Time Performing |
|-----------------|----------|------|-----------------|
| Before Quantize | 70.98    | 1.18 | 41.29           |
| After Quantize  | 64.29    | 1.42 | 9.6             |

## PerChannelMovingAverageMinMax Quantization for weights and MovingAveragePerTensorQuantization for Activation

Results:

| Model           | Accuracy | Loss | Time Performing |
|-----------------|----------|------|-----------------|
| Before Quantize | 70.98    | 1.16 | 47.18           |
| After Quantize  | 59.82    | 1.73 | 10.52           |

# Conclusions

After Quantization we can see that time of evaluating and net inference is much more quicker (around 4x).
But in most cases we are loosing too much precision (12%). In case with PerTensor Quantization for Weights our results
were the best what is surprising because method perChannel should get the best results, but it could result of too 
big focusing on calibration data (quantization overfitting) and could be caused by our calibration set was not representable enough


