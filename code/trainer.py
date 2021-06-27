from torchaudio.transforms import Resample, MelSpectrogram, MFCC
from torchaudio.datasets import SPEECHCOMMANDS
from accelerate import Accelerator
from torch.utils.data import DataLoader
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from utils import collate_fn, mixup
from tqdm import tqdm
from functools import partial
import wandb
import os
from torchmetrics import F1
from torchvision.utils import save_image
from adamp import AdamP
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from augmentations import FrequencyMasking, TimeMasking
from transformers import Wav2Vec2FeatureExtractor

class Trainer():
    def __init__(self, args, model):

        if args.mode == 'train':
            wandb.init(project = 'speech_commands', name = args.run_name)

        self.args = args
        self.accelerator = Accelerator(fp16 = args.use_fp16)
        self.device = self.accelerator.device
        self.model = model
        self.best_f1 = 0

        feature_extractor = None
        if args.model == 'wav2vec2':
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

        if args.swa:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
            0.1 * averaged_model_parameter + 0.9 * model_parameter
            self.swa_model = AveragedModel(model, avg_fn=ema_avg)

        if self.args.mode == 'analysis':
            loaded_ckpt = torch.load(os.path.join(args.ckpt_path, args.run_name, 'best.pth'))

            if args.swa:
                self.model = AveragedModel(self.model)

            self.model.load_state_dict(loaded_ckpt['state_dict'])

        self.model.to(self.device)

        phases = ['train', 'val']

        datasets = {'train': SPEECHCOMMANDS(root = args.data_path, subset = 'training'), 
                    'val': SPEECHCOMMANDS(root = args.data_path, subset = 'validation')}

        # labels = sorted(list(set(datapoint[2] for datapoint in datasets['train'])))
        labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 
                'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
                'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

        self.tfm = self.get_tfm()

        self.dataloaders = {phase: DataLoader(datasets[phase], batch_size = args.batch_size, collate_fn = partial(collate_fn, args = args, labels_list = labels, transform = self.tfm[phase], feature_extractor = feature_extractor), 
                        num_workers = 10, pin_memory = True, shuffle = True if phase == 'train' else False) for phase in phases}


        self.criterion = nn.CrossEntropyLoss(reduction = 'none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience = 3, factor = 0.1)

        if args.cyclelr:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr = 1e-4, max_lr = 1e-3, step_size_up=len(self.dataloaders['train'])/2, cycle_momentum=False)


        self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val'] = self.accelerator.prepare(self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val'])

    def get_tfm(self):

        
        tfm = {'train': [Resample(orig_freq=16000, new_freq=self.args.sample_rate)],
        'val': [Resample(orig_freq=16000, new_freq=self.args.sample_rate)]
        }

        if self.args.input_type == 'melspec':
            tfm['train'].append(MelSpectrogram(sample_rate = self.args.sample_rate, n_fft = 400, hop_length = 200, n_mels = 128))   # n_ftt creates (n_ftt // 2 + 1) bins
            tfm['val'].append(MelSpectrogram(sample_rate = self.args.sample_rate, n_fft = 400, hop_length = 200, n_mels = 128))
        elif self.args.input_type == 'mfcc':
            tfm['train'].append(MFCC(sample_rate = self.args.sample_rate, n_mfcc = 40, dct_type = 2))
            tfm['val'].append(MFCC(sample_rate = self.args.sample_rate, n_mfcc = 40, dct_type = 2))

        if self.args.do_aug:
            tfm['train'].append(FrequencyMasking(freq_mask_param = 128, iid_masks = True, prob = 0.5))
            tfm['train'].append(TimeMasking(time_mask_param = 201, iid_masks = True, prob = 0.5))

        tfm['train'] = nn.Sequential(*tfm['train'])
        tfm['val'] = nn.Sequential(*tfm['val'])

        return tfm


    def forward(self, audio, targets, model, phase):
        audio = audio.to(self.device)
        targets = targets.to(self.device)

        if phase == 'train' and self.args.mixup:
            audio, shuffled_targets, lam = mixup(audio, targets, alpha = 1.0)

        outputs = model(audio)

        if self.args.model == 'wav2vec2':
            outputs = outputs.logits

        if phase == 'train' and self.args.mixup:
            loss = lam * self.criterion(outputs.squeeze(1), targets) + (1 - lam) * self.criterion(outputs.squeeze(1), shuffled_targets)
        else:
            loss = self.criterion(outputs.squeeze(1), targets)

        return loss, outputs

    def update_student(self, loss, phase):
        if phase == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.args.cyclelr:
                self.scheduler.step()
                lr = self.scheduler.get_lr()
    
                wandb.log({'train_step_loss': loss, 'learning_rate': lr[0]})
            else:
                wandb.log({'train_step_loss': loss})

        elif phase == 'val' and self.args.mode != 'analysis':
            wandb.log({'val_step_loss': loss})

    def iterate(self, epoch, phase):

        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        f1_metric = F1(num_classes = 35, average = 'micro')

        if phase == 'train':
            self.model.train()
        elif phase == 'val':
            if self.args.swa:
                self.swa_model.eval()
            else:
                self.model.eval()

        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        loss_list = []
        target_list = []
        pred_list = []

        self.optimizer.zero_grad()
        for itr, batch in tqdm(enumerate(dataloader), total = total_batches): 
            audio, targets = batch

            if self.args.swa and phase == 'val':
                loss, outputs = self.forward(audio, targets, self.swa_model, phase)
            else:
                loss, outputs = self.forward(audio, targets, self.model, phase)

            loss_list.append(loss.reshape(targets.shape[0], -1).mean(-1).detach().cpu())

            loss = loss.mean()

            self.update_student(loss, phase)

            running_loss += loss.item()
            pred_list.append(outputs.argmax(-1).squeeze().detach().cpu())
            target_list.append(targets.detach().cpu())

            # if itr == 200:
            #     break

        epoch_loss = running_loss / total_batches
        f1_score = f1_metric(torch.cat(pred_list), torch.cat(target_list))

        print("Loss: %0.4f | F1 score: %0.4f" % (epoch_loss, f1_score))


        loss_list = torch.cat(loss_list)
        top_losses, top_indices = torch.topk(loss_list, self.args.num_audio_save, largest = True)

        if self.args.mode == 'train':
            wandb.log({f'{phase}_epoch_loss': epoch_loss, f'{phase}_f1': f1_score})

        if self.args.mode ==  'train':
            if self.args.swa and phase == 'train': 
                self.swa_model.update_parameters(self.model)
                torch.optim.swa_utils.update_bn(self.dataloaders[phase], self.swa_model, device = 'cuda')

        torch.cuda.empty_cache()

        return epoch_loss, top_losses, top_indices, f1_score

    def start_training(self):

        for epoch in range(self.args.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict() if not self.args.swa else self.swa_model.state_dict(),
            }
            with torch.no_grad():
                val_loss, _, _, f1_score = self.iterate(epoch, "val")

            if not self.args.cyclelr:
                self.scheduler.step(val_loss)

            if f1_score > self.best_f1:
                print("******** New optimal found, saving state ********")
                state["best_f1"] = self.best_f1 = f1_score
                os.makedirs(f"/SSD1TB/audio/checkpoints/{self.args.run_name}", exist_ok = True)
                torch.save(state, f"/SSD1TB/audio/checkpoints/{self.args.run_name}/best.pth")


class Evaluator(object):
    '''This class takes care of evaluation of our model'''
    def __init__(self, args, model):

        self.args = args
        phases = ["test"]
 
        self.accelerator = Accelerator(fp16 = args.use_fp16)
        self.device = self.accelerator.device

        feature_extractor = None
        if args.model == 'wav2vec2':
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

        self.net = model
        loaded_ckpt = torch.load(os.path.join(args.ckpt_path, args.run_name, 'best.pth'))

        if args.swa:
            self.net = AveragedModel(self.net)

        self.net.load_state_dict(loaded_ckpt['state_dict'])

        self.net.to(self.device)

        datasets = {'test': SPEECHCOMMANDS(root = args.data_path, subset = 'testing')}

        # labels = sorted(list(set(datapoint[2] for datapoint in datasets['train'])))
        labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 
                'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
                'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

        self.tfm = self.get_tfm()

        self.dataloaders = {phase: DataLoader(datasets[phase], batch_size = args.batch_size, collate_fn = partial(collate_fn, args = args, labels_list = labels, transform = self.tfm[phase], feature_extractor = feature_extractor), 
                        num_workers = 10, pin_memory = True, shuffle = True if phase == 'train' else False) for phase in phases}

        self.net, self.dataloaders['test'] = self.accelerator.prepare(self.net, self.dataloaders['test'])

    def get_tfm(self):

        
        tfm = {'test': [Resample(orig_freq=16000, new_freq=self.args.sample_rate)],
        }

        if self.args.input_type == 'melspec':
            tfm['test'].append(MelSpectrogram(sample_rate = self.args.sample_rate, n_fft = 400, hop_length = 200, n_mels = 128))   # n_ftt creates (n_ftt // 2 + 1) bins
        elif self.args.input_type == 'mfcc':
            tfm['test'].append(MFCC(sample_rate = self.args.sample_rate, n_mfcc = 40, dct_type = 2))



        tfm['test'] = nn.Sequential(*tfm['test'])


        return tfm
        
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)

        if self.args.model == 'wav2vec2':
            outputs = outputs.logits

        return outputs

    def iterate(self, phase):

        start = time.strftime("%H:%M:%S")
        print(f"Phase: {phase} | ⏰: {start}")

        self.net.eval()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        f1_metric = F1(num_classes = 35, average = 'micro')

        pred_list = []
        target_list = []

        for itr, batch in tqdm(enumerate(dataloader), total = total_batches): 
            images, targets = batch
            outputs = self.forward(images, targets)

            pred_list.append(outputs.argmax(-1).squeeze().detach().cpu())
            target_list.append(targets.detach().cpu())

        f1_score = f1_metric(torch.cat(pred_list), torch.cat(target_list))
        print("F1 score: %0.4f" % (f1_score))        

        torch.cuda.empty_cache()

        return f1_score

    def start(self):

        with torch.no_grad():
            f1_score = self.iterate('test')

        if not self.args.review:
            os.makedirs('/SSD1TB/audio/test_results', exist_ok = True)
            with open(f'/SSD1TB/audio/test_results/{self.args.run_name}.txt', 'w') as f:
                f.write(f'Test F1 score: {np.round(f1_score, 4)}')