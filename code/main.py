from __future__ import print_function

from model import TRANSFORMER_ENCODER
from miscc.config import cfg, cfg_from_file
from miscc.utils import collapse_dirs, mv_to_paths
from miscc.metrics import compute_ppl
from datasets import TextDataset, ImageFolderDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as transforms

from nltk.tokenize import RegexpTokenizer
from transformers import GPT2Tokenizer

from tqdm import tqdm
import pytorch_fid.fid_score

import datetime
import re

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

global globaltimestamp 
global latentSpaceMode
global string_of_tokens
globaltimestamp = str(datetime.datetime.now())
latentSpaceMode="1"
string_of_tokens="unknown"
#makeWords = True
#renderIt = True
debug = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--text_encoder_type', type=str.casefold, default = 'rnn' )
    # add latentSpaceMode
    parser.add_argument('--latentSpaceMode', dest='latentSpaceMode', type=str, default='1')
    parser.add_argument('--frames', dest='frames', type=int, default='10')
    parser.add_argument('--AR', dest='AR', type=bool, default=False)
    parser.add_argument('--makeWords', dest='makeWords', type=bool, default=True)
    parser.add_argument('--render', dest='renderIt', type=bool, default=True)
    args = parser.parse_args()
    return args


def gen_example(wordtoix, text_encoder_type, algo, frames, makeWords, renderIt):
    '''generate images from example sentences'''
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    frames = frames
    text_encoder_type = text_encoder_type.casefold()
    if text_encoder_type not in ( 'rnn', 'transformer' ):
      raise ValueError( 'Unsupported text_encoder_type' )
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    if text_encoder_type == 'rnn':
                      print ("text_encoder_type: rrn ")
                      tokenizer = RegexpTokenizer(r'\w+')
                      tokens = tokenizer.tokenize( sent.lower() )
                    elif text_encoder_type == 'transformer':
                      print ("text_encoder_type: transformer ")
                      tokenizer = GPT2Tokenizer.from_pretrained( TRANSFORMER_ENCODER )
                      tokens = tokenizer.tokenize( sent )
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    string_of_tokens=""
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                            print ("Found Token: ")
                            print ( t )
                            print ("index")
                            print(wordtoix[t])
                            string_of_tokens = string_of_tokens + t
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices, string_of_tokens, globaltimestamp, frames, string_of_tokens]
    if makeWords:
        algo.gen_word_feature_tensor(data_dic)
    if renderIt:
        algo.gen_example(data_dic)

if __name__ == "__main__":
    args = parse_args()
    print (globaltimestamp ) #= str(datetime.datetime.now())
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    frames = args.frames
    makeWords = args.makeWords
    renderIt = args.renderIt
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    print ("TextDataset cfg.DATA_DIR")
    print (cfg.DATA_DIR)
    dataset = TextDataset(cfg.DATA_DIR, args.text_encoder_type, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate

    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset.text_encoder_type, globaltimestamp, latentSpaceMode, string_of_tokens)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        print( '\nTraining...\n+++++++++++' )
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
    else:
        # generate images from pre-extracted embeddings
        if not cfg.B_VALIDATION:
            # generate images for customized captions
            print( '\nRunning on example captions...\n++++++++++++++++++++++++++++++' )
            if debug:
                print (dataset.wordtoix)
            #frames = 3 
            root_dir_g = gen_example(dataset.wordtoix, dataset.text_encoder_type, algo, frames, makeWords, renderIt)
            end_t = time.time()
            print('Total time for running on example captions:', end_t - start_t)
        else:
            # generate images for the whole valid dataset
            print( '\nValidating...\n+++++++++++++' )
            root_dir_g = algo.sampling(split_dir)
            end_t = time.time()
            print('Total time for validation:', end_t - start_t)
            print()

            # GAN Metrics
            if cfg.B_FID or cfg.B_PPL:  # or cfg.B_IS:
                device = torch.device( 'cuda' if (torch.cuda.is_available()) else 'cpu' )
                num_metrics = 0
                final_dir_g = str( Path( root_dir_g ).parent/'metrics' )
                # compute FID
                if cfg.B_FID:
                    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                    print( 'Computing FID...' )
                    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                    orig_paths_g, final_paths_g = collapse_dirs( root_dir_g, final_dir_g )
                    # -- #
                    data_dir_r = '%s/CUB_200_2011' % dataset.data_dir if dataset.bbox is not None else dataset.data_dir
                    root_dir_r = os.path.join( data_dir_r, 'images' )
                    final_dir_r = os.path.join( root_dir_r, f'{imsize}x{imsize}' )
                    orig_paths_r, final_paths_r = collapse_dirs( root_dir_r, final_dir_r, copy = True, ext = '.jpg' )
                    dataset_rsz = ImageFolderDataset( img_paths = final_paths_r,
                                                      transform = image_transform,  # transforms.Compose([transforms.Resize((imsize, imsize,))]),
                                                      save_transformed = True )
                    dataloader_rsz = torch.utils.data.DataLoader( dataset_rsz, batch_size = cfg.TRAIN.BATCH_SIZE,
                                                                  drop_last = False, shuffle = False, num_workers = int(cfg.WORKERS) )
                    dl_itr = iter( dataloader_rsz )
                    print( f'Resizing real images to that of generated images and then saving into {final_dir_r}' )
                    for batch_itr in tqdm( range( len( dataloader_rsz ) ) ):
                        next( dl_itr )
                    # -- #
                    print( f'Number of generated images to be used in FID calculation: {len( final_paths_g )}' )
                    print( f'Number of real images to be used in FID calculation: {len( final_paths_r )}' )
                    fid_value = pytorch_fid.fid_score.calculate_fid_given_paths( paths = [ final_dir_g, final_dir_r ],
                                                                                           batch_size = 50,
                                                                                           device = device,
                                                                                           dims = 2048 )
                    mv_to_paths( final_paths_g, orig_paths_g )
                    with open( os.path.join( final_dir_g, 'metrics.txt' ), 'w' if num_metrics == 0 else 'a' ) as f:
                        f.write( 'Frechet Inception Distance (FID): {:f}\n'.format( fid_value ) )
                        f.write( 'Root Directories for Datasets used in Calculation: {}, {}\n\n'.format( root_dir_g, root_dir_r ) )
                        num_metrics += 1
                    print( '---> Frechet Inception Distance (FID): {:f}\n'.format( fid_value ) )
                # compute PPL
                if cfg.B_PPL:
                    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                    print( 'Computing PPL...' )
                    print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                    print( f'Number of generated images to be used in PPL calculation: ~{cfg.PPL_NUM_SAMPLES}' )
                    ppl_value = compute_ppl( algo, space = 'smart', num_samples = cfg.PPL_NUM_SAMPLES, eps = 1e-4, net = 'vgg' )
                    with open( os.path.join( final_dir_g, 'metrics.txt' ), 'w' if num_metrics == 0 else 'a' ) as f:
                        f.write( 'Perceptual Path Length (PPL): {:f}\n'.format( ppl_value ) )
                        f.write( 'Root Directories for Datasets used in Calculation: {}\n\n'.format( root_dir_g ) )
                        num_metrics += 1
                    print( '---> Perceptual Path Length (PPL): {:f}\n'.format( ppl_value ) )
                # # compute IS         # NOTE: See README.md for IS
                # if cfg.B_IS:
                #     print( '\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                #     print( 'Computing IS...' )
                #     print( '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' )
                #     print( f'Number of generated images to be used in IS calculation: {len( final_paths_g )}' )
                #     # exec(open("/home/sid/repos/ml-repos/inception-score-pytorch/inception_score.py").read())
                #     exec(open("/home/sid/repos/ml-repos/StackGAN-inception-model/inception_score.py").read())
                #     with open( os.path.join( final_dir_g, 'metrics.txt' ), 'a' ) as f:
                #         f.write( 'Inception Score (IS): {:f}\n'.format( is_value ) )
                #         f.write( 'Root Directories for Datasets used in Calculation: {}\n'.format( root_dir_g ) )
                #     print( '---> Inception Score (IS): {:f}\n'.format( is_value ) )
