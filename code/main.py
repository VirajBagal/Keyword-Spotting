import argparse
from pytorch_lightning.utilities.seed import seed_everything
from models import M5, count_parameters, ExtendedEfficientNet, Wav2Vec2ForSpeechClassification
from trainer import Trainer, Evaluator
from torchvision.models import resnet18
import wandb
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required = True, help='path to main checkpoints dir')
    parser.add_argument("--data_path", default='../', help='path to main checkpoints dir')
    parser.add_argument("--run_name", type=str, required = True)
    parser.add_argument("--model", type=str, default = 'resnet18', help = 'which model to use')
    parser.add_argument("--input_type", type=str, default = 'mfcc', help = 'which input type for the model')
    parser.add_argument("--mixup", action = 'store_true', help='use mixup for augmentation')
    parser.add_argument("--num_epochs", default=30, type=int, help="num_epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size")
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--num_audio_save", default=100, type=int, help="Number of images to save of validation")
    parser.add_argument("--lr", default=1e-4, type=float, help='learning_rate')
    parser.add_argument("--mode", default = 'train', help='use one among "train, test, analysis')
    parser.add_argument("--use_fp16", action = 'store_true', help='use fp16 training')
    parser.add_argument("--swa", action = 'store_true', help='use SWA for potential generalization')
    parser.add_argument("--do_aug", action = 'store_true', help='do data augmentation')
    parser.add_argument("--cyclelr", action = 'store_true', help='use cyclic lr')
    parser.add_argument('--review', action='store_true', help='use this to NOT save anything during review run')

    args = parser.parse_args()

    seed_everything(seed = 42)  # for reproducible results

    if args.model == 'M5':
        model = M5()
    elif 'resnet' in args.model:
        model = resnet18(pretrained = True)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, 35)
    elif 'efficientnet' in args.model:
        model = ExtendedEfficientNet(args)
    else:
        model = Wav2Vec2ForSpeechClassification.from_pretrained('facebook/wav2vec2-base-960h')
        model.freeze_feature_extractor()   # very important to freeze the feature extractor

    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    if args.mode == 'train':
        trainer_class = Trainer(args, model)
        trainer_class.start_training()
    elif args.mode == 'test':
        evaluator = Evaluator(args, model)
        evaluator.start()

