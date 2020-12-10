#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
import os
from utils import *
from options import MainOptions
from data_loader import prepare_dataloader
#from architectures import create_architecture
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random, cv2, argparse
from hparams import hparams, get_image_list


# In[ ]:


from architectures.base_architecture import BaseArchitecture
from architectures.monodepth_architecture import MonoDepthArchitecture


# In[ ]:


# ----- The following cell is taken from Wav2Lip

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

def save_sample_images(x, g_left, gt_left, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g_left = (g_left.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt_left = (gt_left.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g_left, gt_left), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g_left):
    g_left = g_left[:, :, :, g_left.size(3)//2:]
    g_left = torch.cat([g_left[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g_left)
    y = torch.ones(g_left.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

#------- End of Wav2Lip cell


# In[ ]:


def validate_cycle(global_epoch): # M
    model_cycle.to_test() # M
    disparities = np.zeros((val_n_img, 256, 512), dtype=np.float32)
    model_cycle.set_new_loss_item(global_epoch, train=False) # M
    torch.set_grad_enabled(False)

    for i, data in enumerate(val_loader):
        model_cycle.set_input(data) # M
        model_cycle.forward() # M
        model_cycle.add_running_loss_val(global_epoch) # M

    torch.set_grad_enabled(True)
    model_cycle.make_running_loss(global_epoch, val_n_img, train=False) # M
    return None


# In[ ]:


def train(args, device, model_Wav2Lip, train_data_loader_Left, test_data_loader_Left, optimizer, checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    n_img, loader = prepare_dataloader(args, 'train') # CycleGAN
    val_n_img, val_loader = prepare_dataloader(args, 'val') # CycleGAN
    model_cycle = MonoDepthArchitecture(args) # Modified
    model_cycle.set_data_loader(loader) # Modified
    
    global global_step, global_epoch # Wav2Lip
    resumed_step = global_step # Wav2Lip
    
    if not args.resume: # CycleGAN
        best_val_loss = float('Inf') # CycleGAN
        validate_cycle(-1) # # CycleGAN
        pre_validation_update(model_cycle.losses[-1]['val']) # Modified
    else:
        best_val_loss = min([model_cycle.losses[epoch]['val']['G'] for epoch in model_cycle.losses.keys()]) # Modified

    running_val_loss = 0.0
    
    while global_epoch < nepochs:
        # Cycle GAN
        c_time = time.time()
        model_cycle.to_train() # Modified
        model_cycle.set_new_loss_item(global_epoch) # Modified

        model_cycle.run_epoch(global_epoch, n_img) # Modified
        validate_cycle(global_epoch) # M
        print_epoch_update(global_epoch, time.time() - c_time, model_cycle.losses) # Modified

        # Make a checkpoint
        running_val_loss = model_cycle.losses[global_epoch]['val']['G'] # Modified
        is_best = running_val_loss < best_val_loss
        
        if is_best:
            best_val_loss = running_val_loss
            
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader_Left))
        
        for step, (x, indiv_mels, mel, gt_left) in prog_bar: # M
            model_Wav2Lip.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt_left = gt_left.to(device) # Modified

            g_left = model_Wav2Lip(indiv_mels, x) # Modified

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g_left)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g_left, gt_left) + best_val_loss * (recon_loss(g_left, gt_left)) # Modified 

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g_left, gt_left, global_step, checkpoint_dir)

            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model_Wav2Lip, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader_Left, global_step, device, model_Wav2Lip, checkpoint_dir, best_val_loss)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.01) # without image GAN a lesser weight is sufficient

            prog_bar.set_description('L1: {}, Sync Loss: {}'.format(running_l1_loss / (step + 1),
                                                                    running_sync_loss / (step + 1)))
            
        model_cycle.save_checkpoint(global_epoch, is_best, best_val_loss)

        global_epoch += 1
        
    print('Finished Training. Best validation loss:\t{:.3f}'.format(best_val_loss))
    model_cycle.save_networks('final')
    
    if running_val_loss != best_val_loss:
        model_cycle.save_best_networks()
        
    model_cycle.save_losses()


# In[ ]:


# Wav2Lip

def save_checkpoint(model_Wav2Lip, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model_Wav2Lip.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model_Wav2Lip, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model_Wav2Lip.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model_Wav2Lip


# In[ ]:


def eval_model(test_data_loader_Left, global_step, device, model_Wav2Lip, checkpoint_dir, best_val_loss):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt_left in test_data_loader:
            step += 1
            model_Wav2Lip.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt_left = gt_left.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            g_left = model_Wav2Lip(indiv_mels, x)

            sync_loss = get_sync_loss(mel, g_left)
            
            l1loss = recon_loss(g_left, gt_left) + (best_val_loss * recon_loss(g_left, gt_left))

            sync_losses.append(sync_loss.item())
            recon_losses.append(l1loss.item())

            if step > eval_steps: 
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                print('L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))
                return averaged_sync_loss


# In[ ]:


### This cell (test function is taken as is from Wav2Lip)
def test(args):
    """ Function to test the architecture by saving disparities to the output directory
    """
    # Since it is clear post-processing is better in all runs I have done, I will only
    # save post-processed results. Unless explicitly stated otherwise.
    # Also for Pilzer, the disparities are already post-processed by their own FuseNet.
    do_post_processing = args.postprocessing and 'pilzer' not in args.architecture

    input_height = args.input_height
    input_width = args.input_width

    output_directory = args.output_dir
    n_img, test_loader = prepare_dataloader(args, 'test')

    model_cycle = MonoDepthArchitecture(args)
    which_model = 'final' if args.load_final else 'best'
    model_cycle.load_networks(which_model)
    model_cycle.to_test()

    disparities = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    inference_time = 0.0

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 100 == 0 and i != 0:
                print('Testing... Now at image: {}'.format(i))

            t_start = time.time()
            # Do a forward pass
            disps = model_cycle.fit(data)
            # Some architectures output a single disparity, not a tuple of 4 disparities.
            disps = disps[0][:, 0, :, :] if isinstance(disps, tuple) else disps.squeeze()

            if do_post_processing:
                disparities[i] = post_process_disparity(disps.cpu().numpy())
            else:
                disp = disps.unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
            t_end = time.time()
            inference_time += (t_end - t_start)

    if args.test_time:
        test_time_message = 'Inference took {:.4f} seconds. That is {:.2f} imgs/s or {:.6f} s/img.'
        print(test_time_message.format(inference_time, (n_img / inference_time), 1.0 / (n_img / inference_time)))

    disp_file_name = 'disparities_{}_{}.npy'.format(args.dataset, model_cycle.name)
    full_disp_path = os.path.join(output_directory, disp_file_name)

    if os.path.exists(full_disp_path):
        print('Overwriting disparities at {}...'.format(full_disp_path))
    np.save(full_disp_path, disparities)
    print('Finished Testing')
    
## End of Wav2Lip test


# In[ ]:



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code to train the Cycle GAN and Wav2Lip togethor')
    parser.add_argument('--architecture',       type=str,   help='The architecture from which exp to run.', default='monodepth')
    parser.add_argument('--mode',               type=str,   help='mode: train or test', default='train')
    parser.add_argument('--split',              type=str,   help='data split, kitti or eigen', default='eigen')
    parser.add_argument('--dataset',            type=str,   help='dataset to train on kitti or cityscapes or both', default='kitti')
    parser.add_argument('--data_dir',           type=str,   help='path to the data directory', required=True)
    parser.add_argument('--output_dir',         type=str,   help='where to save the disparities', default='output/')
    parser.add_argument('--model_dir',          type=str,   help='path to the trained models', default='saved_models/')
    parser.add_argument('--model_name',         type=str,   help='model name', default='mono-depth')
    parser.add_argument('--input_height',       type=int,   help='input height', default=256)
    parser.add_argument('--input_width',        type=int,   help='input width', default=512)
    parser.add_argument('--generator_model',    type=str,   help='encoder architecture: [resnet50_pilzer]', default='resnet50_pilzer')
    parser.add_argument('--discriminator_model',type=str,   help='discrininator architecture: [simple]', default='simple')
    parser.add_argument('--pretrained',         type=boolstr,  help='use weights of pretrained model', default=False)
    parser.add_argument('--input_channels',     type=int,   help='Number of channels in input tensor', default=3)
    parser.add_argument('--disc_input_channels',type=int,   help='Number of channels in input tensor for discriminator', default=3)
    parser.add_argument('--num_disp_channels',  type=int,   help='Number of disparity channels', default=2)
    parser.add_argument('--device',             type=str,   help='choose cpu or cuda:0 device"', default='cuda:0')
    parser.add_argument('--use_multiple_gpu',   type=boolstr,  help='whether to use multiple GPUs', default=False)
    parser.add_argument('--num_threads',        type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--epochs',             type=int,   help='number of total epochs to run', default=50)
    parser.add_argument('--learning_rate',      type=float, help='initial learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--adjust_lr',          type=boolstr,  help='apply learning rate decay or not', default=False)
    parser.add_argument('--optimizer',          type=str,   help='Optimizer to use [adam | rmsprop]', default='adam')
    parser.add_argument('--batch_size',         type=int,   help='mini-batch size (default: 8)', default=8)
    parser.add_argument('--generator_loss',     type=str,   help='generator loss [ monodepth | cycled_adversarial | l1 ]', default='monodepth')
    parser.add_argument('--discriminator_loss', type=str,   help='generator loss [ vanilla | ls ]', default='ls')
    parser.add_argument('--img_loss_w',         type=float, help='image reconstruction weight', default=1.0)
    parser.add_argument('--lr_loss_w',          type=float, help='left-right consistency weight', default=1.0)
    parser.add_argument('--alpha_image_loss_w', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--which_ssim',         type=str,   help='use either Godard SSIM or Gaussian [ godard | gauss ]', default='godard')
    parser.add_argument('--ssim_window_size',   type=int,   help='when using Gaussian SSIM, size of window', default=11)
    parser.add_argument('--disp_grad_loss_w',   type=float, help='disparity smoothness weight', default=0.1)
    parser.add_argument('--occl_loss_w',        type=float, help='Occlusion loss weight', default=0.0)
    parser.add_argument('--discriminator_w',    type=float, help='discriminator loss weight', default=0.5)
    parser.add_argument('--do_augmentation',    type=boolstr,  help='do augmentation of images or not', default=True)
    parser.add_argument('--augment_parameters', type=str,   help='lowest and highest values for gamma, brightness and color respectively',
                            default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2])
    parser.add_argument('--norm_layer',         type=str,   help='defines if a normalization layer is used', default='')
    parser.add_argument('--num_disps',          type=int,   help='Number of predicted disparity maps', default=4)
    parser.add_argument('--wgan_critics_num',   type=int,   help='number of critics in the WGAN architecture', default=5)

    parser.add_argument('--train_ratio',        type=float, help='How much of the training data to use', default=1.0)
    parser.add_argument('--resume',             type=str,   help='path to latest checkpoint (default: none)', default='')
    parser.add_argument('--resume_regime',      type=str,   help='back-passes of the G, D resp. default: 0,0', default=[0, 0])

    parser.add_argument('--postprocessing',     type=boolstr,  help='Do post-processing on depth maps', default=True)
    parser.add_argument('--load_final',         type=boolstr,  help='Load final or best trained model', default=True)
    parser.add_argument('--test_time',          type=boolstr, help='Print the time of inference', default=False)
    
    # Wav2Lip args options
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset + Our collected Dataset either left or right stereo", required=True, type=str)
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

    args = parser.parse_args()
    global_step = 0
    global_epoch = 0
    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}'.format(use_cuda))
    syncnet_T = 5
    syncnet_mel_step_size = 16
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset_left = Dataset('train') # Left stereo image training dataset
    test_dataset_left = Dataset('val') # Left Stereo image validation dataset

    train_data_loader_Left = data_utils.DataLoader(train_dataset_left, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader_Left = data_utils.DataLoader(test_dataset_left, batch_size=hparams.batch_size,num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    model_Wav2Lip = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model_Wav2Lip.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model_Wav2Lip.parameters() if p.requires_grad], lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model_Wav2Lip, optimizer, reset_optimizer=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    if args.mode == 'train':
        train(args, device, model_Wav2Lip, train_data_loader_Left, test_data_loader_Left, optimizer, checkpoint_dir=checkpoint_dir, checkpoint_interval=hparams.checkpoint_interval, nepochs=hparams.nepochs)
    elif args.mode == 'test':
        test(args) # Only Cycle GAN 
    elif args.mode == 'verify-data':
        from utils.reduce_image_set import check_if_all_images_are_present
        check_if_all_images_are_present('kitti', args.data_dir)
        check_if_all_images_are_present('eigen', args.data_dir)
        check_if_all_images_are_present('cityscapes', args.data_dir)

