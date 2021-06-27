# Keyword-Spotting


![alt text](https://github.com/VirajBagal/Keyword-Spotting/keyword_spotting.jpeg?raw=true)


In this project, we do an ablation study of existing audio processing models for 'keyword spotting' using the Speech Commands dataset. The dataset has 65,000 one second long utterances of 30 short words in English done by thousands of people. 

Evaluation curves for all experiments can be found here: 

https://wandb.ai/vbagal/speech_commands


# How to reproduce the results?

Clone the repo and follow these steps.

- Download dataset using following command:
```
python download_dataset.py --save_dir <path_to_save_the_dataset>
```

- To train the baseline:
```
python main.py --run_name <some_name> --data_path <path_where_dataset_is_saved> --model M5 --input_type waveform --batch_size 512
```

To try out different approaches, please use the keywords like model, input_type, mixup, do_aug, cyclelr. For saving weights, new directory called checkpoints/<some_name> is created. 

- To test the model:
```
python main.py --run_name <same_name_as_before> --data_path <path_where_dataset_is_saved> --ckpt_path <path_to_checkpoint> --model M5 --input_type waveform --batch_size 512 --mode test
```

# Comparison of different approaches

| Method | Raw Audio | Mel Spec | MFCC | Test F1 |
| :---: | :---: | :---: | :---: | :---: | 
| M5 | :heavy_check_mark: | :x: | :x: | 0.8636 |
| Resnet-18 | :x: | :heavy_check_mark: | :x: | 0.9246 |
| Resnet-18 | :x: | :x: | :heavy_check_mark: | 0.9522 |
| EfficientNet-B2 | :x: | :x: | :heavy_check_mark: | 0.9507 | 
| EfficientNet-B4 | :x: | :x: | :heavy_check_mark: | 0.9558 | 
| Wav2Vec 2.0 | :heavy_check_mark: | :x: | :x: | 0.9746 | 

# Ablation on ResNet-18


| Input | Mixup | Mask Augs | Test F1 |
| :---: | :---: | :---: | :---: |
| Mel Spec | :x: | :x: | 0.9246 |
| Mel Spec | :heavy_check_mark: | :x: | 0.9283 |
| MFCC | :heavy_check_mark: | :x: | 0.9522 |
| MFCC | :x: | :heavy_check_mark: | 0.9510 |
| MFCC | :heavy_check_mark: | :heavy_check_mark: | 0.9410 |

