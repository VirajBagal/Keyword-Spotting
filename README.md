# Keyword-Spotting
In this project, we do an ablation study of existing audio processing models for 'keyword spotting' using the Speech Commands dataset. The dataset has 65,000 1 second long utterances of 30 short words in English done by thousands of people. 

# Comparison of different approaches

| Method | Raw Audio | Mel Spec | MFCC | Test F1 |
| --- | --- | --- | --- | --- | 
| M5 | :heavy_check_mark: | :x: | :x: | 0.8636 |
| Resnet-18 | :x: | :heavy_check_mark: | :x: | 0.9246 |
| Resnet-18 | :x: | :x: | :heavy_check_mark: | 0.9522 |
| EfficientNet-B2 | :x: | :x: | :heavy_check_mark: | 0.9507 | 
| EfficientNet-B4 | :x: | :x: | :heavy_check_mark: | 0.9558 | 
| Wav2Vec 2.0 | :heavy_check_mark: | :x: | :x: | 0.9710 | 
