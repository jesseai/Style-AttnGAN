from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, G_NET_STYLED
from datasets import prepare_data
from model import TRANSFORMER_ENCODER, RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os

#jesse20210730
import datetime
import re
import json
#essej

import time
import numpy as np
import sys

from transformers import GPT2Model


def interpolateFunction(inTensor,a,b,readBack,f,frames):
  a = 1 - (1/frames) * f
  b = (1/frames) * f
  print (a,b)
  out_words_embs = (inTensor * a) + (readBack * b)
  return out_words_embs


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, text_encoder_type, globaltimestamp, latentSpaceMode, string_of_tokens):
        if cfg.TRAIN.FLAG:
            self.output_dir = output_dir
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.in_globaltimestamp=globaltimestamp
        self.in_string_of_tokens = string_of_tokens
        self.text_encoder_type = text_encoder_type.casefold()
        if self.text_encoder_type not in ( 'rnn', 'transformer' ):
          raise ValueError( 'Unsupported text_encoder_type' )

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('(trainer.py) Load image encoder from:', img_encoder_path)
        image_encoder.eval()
        #print ("m build_models index to word")
        #print (self.ixtoword)
        if self.text_encoder_type == 'rnn':
            print ("self text encoder rnn")
            print ("m build_models index to word")
            print (self.ixtoword)
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM, globaltimestamp=self.in_globaltimestamp, string_of_tokens=self.in_string_of_tokens)
        elif self.text_encoder_type == 'transformer':
            text_encoder = GPT2Model.from_pretrained( TRANSFORMER_ENCODER )
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('(trainer.py)Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        elif cfg.GAN.B_STYLEGEN:
            netG = G_NET_STYLED()
            if cfg.GAN.B_STYLEDISC:
                from model import D_NET_STYLED64, D_NET_STYLED128, D_NET_STYLED256
                if cfg.TREE.BRANCH_NUM > 0:
                    netsD.append(D_NET_STYLED64())
                if cfg.TREE.BRANCH_NUM > 1:
                    netsD.append(D_NET_STYLED128())
                if cfg.TREE.BRANCH_NUM > 2:
                    netsD.append(D_NET_STYLED256())
                # TODO: if cfg.TREE.BRANCH_NUM > 3:
            else:
                from model import D_NET64, D_NET128, D_NET256 
                if cfg.TREE.BRANCH_NUM > 0:
                    netsD.append(D_NET64())
                if cfg.TREE.BRANCH_NUM > 1:
                    netsD.append(D_NET128())
                if cfg.TREE.BRANCH_NUM > 2:
                    netsD.append(D_NET256())
                # TODO: if cfg.TREE.BRANCH_NUM > 3:
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
            netG.apply(weights_init)
            # print(netG)
            for i in range(len(netsD)):
                netsD[i].apply(weights_init)
                # print(netsD[i])
        print(netG.__class__)
        for i in netsD: print( i.__class__ )
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            if cfg.GAN.B_STYLEGEN:
                netG.w_ewma = state_dict[ 'w_ewma' ]
                if cfg.CUDA:
                    netG.w_ewma = netG.w_ewma.to( 'cuda:' + str( cfg.GPU_ID ) )
                netG.load_state_dict( state_dict[ 'netG_state_dict' ] )
            else:
                netG.load_state_dict( state_dict )
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def build_models_eval(self, init_func = None):
        print("build_models_eval self.ixtoword exists here")
        #print ( str(captions) )
        # #######################generator########################### #
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        elif cfg.GAN.B_STYLEGEN:
            netG = G_NET_STYLED()
        else:
            netG = G_NET()
            if init_func is not None:
                netG.apply(init_func)
        # print( netG.__class__ )
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images
        try:
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = torch.load(model_dir, map_location = lambda storage, loc: storage)
        except:
            msg = f'The path for the models cfg.TRAIN.NET_G = {cfg.TRAIN.NET_G} is not found'
            raise ValueError( msg )
        if cfg.GAN.B_STYLEGEN:
            # netG.load_state_dict( state_dict )
            netG.w_ewma = state_dict[ 'w_ewma' ]
            if cfg.CUDA:
                netG.w_ewma = netG.w_ewma.to( 'cuda:' + str( cfg.GPU_ID ) )
            netG.load_state_dict( state_dict[ 'netG_state_dict' ] )
        else:
            netG.load_state_dict( state_dict )
        print('Load G from: ', model_dir)
        netG.cuda()
        netG.eval()

        # ###################text encoder########################### #
        if self.text_encoder_type == 'rnn':
            print("build_models_eval rnn")
            text_encoder = RNN_ENCODER(self.n_words, globaltimestamp=self.in_globaltimestamp, string_of_tokens=self.in_string_of_tokens, nhidden=cfg.TEXT.EMBEDDING_DIM )
        elif self.text_encoder_type == 'transformer':
            text_encoder = GPT2Model.from_pretrained( TRANSFORMER_ENCODER )
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('(trainer.py) : build_models_eval : Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        return text_encoder, netG

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        _pathG = '%s/netG_epoch_%d.pth' % ( self.model_dir, epoch )
        if cfg.GAN.B_STYLEGEN:
            torch.save( {
                'w_ewma': netG.w_ewma,  # .to( 'cpu' ),
                'netG_state_dict': netG.state_dict()
                }, _pathG
            )
        else:
            torch.save( netG.state_dict(), _pathG )
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        if cfg.GAN.B_STYLEGEN:
            netG.eval()
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        if cfg.GAN.B_STYLEGEN:
            netG.train()

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):  # EWMA time-averaging
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                    with open( '%s/D_logs.txt' % ( self.output_dir ), 'a' ) as f:
                        f.write( D_logs + '\n' )
                    with open( '%s/G_logs.txt' % ( self.output_dir ), 'a' ) as f:
                        f.write( G_logs + '\n' )
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if split_dir == 'test':
            split_dir = 'valid'
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images

        # Build and load the generator and text encoder
        text_encoder, netG = self.build_models_eval(init_func = weights_init)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, nz))
            noise = noise.cuda()

        # the path to save generated images
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)

        cnt = 0

        for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break

                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                #######################################################
                # (1) Extract text embeddings
                ######################################################
                print ("(1) Extract text embeddings")
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions, self.in_globaltimestamp )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                print ("(2) Generate fake images")
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d._.png' % (s_tmp, k)
                    im.save(fullpath)
        return save_dir


    def delete_gen_word_feature_tensor(self, data_dic):
        AR = False

        featureVectorArea =1 #this is the Q of elements in the feature vector to deform
        # this is * by frames, for a 2d area

        theAmountOfNoise = 9
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images
        
        # Build and load the generator and text encoder
        print ("(trainer.py) gen_example : Build and load the generator and text encoder")
        text_encoder, netG = self.build_models_eval()
        print ("At this point the self.build_models_eval() generates an RNN model, which then writes a tensor to disk. ")



    def gen_word_feature_tensor(self, data_dic):
        AR = False

        featureVectorArea =1 #this is the Q of elements in the feature vector to deform
        # this is * by frames, for a 2d area

        theAmountOfNoise = 9
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images
        
        # Build and load the generator and text encoder
        print ("(trainer.py) gen_example : Build and load the generator and text encoder")
        text_encoder, netG = self.build_models_eval()
        print ("At this point the self.build_models_eval() generates an RNN model, which then writes a tensor to disk. ")

        # the path to save generated images
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        # print( data_dic.keys() )
        save_dirs = []
        print ("(trainer.py) s_tmp is ")
        print (s_tmp)
        print ("print threshold=10_000")
        torch.set_printoptions(threshold=10_000)

        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            save_dirs.append( save_dir )
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices, string_of_tokens, in_timestamp, frames , string_of_tokens = data_dic[key]
            print ("save_dir")
            print (save_dir)
            print ("string_of_tokens")
            print (string_of_tokens)
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM

            with torch.no_grad():
                captions = Variable(torch.from_numpy(captions))
                cap_lens = Variable(torch.from_numpy(cap_lens))



                captions = captions.cuda()


                cap_lens = cap_lens.cuda()


                print ("captions0")
                print (captions)
                print ("indices0")
                print (cap_lens)
                print ("string_of_tokens")
                print (string_of_tokens)
                print (in_timestamp)
            for i in range(1):  # 16
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    # noise = Variable(torch.FloatTensor(1, nz))
                    noise = noise.cuda()

                #######################################################
                print ("gen_word_feature_tensor Extract text embeddings")
                ######################################################
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden, self.in_globaltimestamp  )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                    print ("(trainer.py) gen_example (1) Extract text embeddings")
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                mask = (captions == 0)





    def gen_example(self, data_dic, frames=10):
        AR = False
        
        featureVectorArea =1 #this is the Q of elements in the feature vector to deform
        # this is * by frames, for a 2d area

        theAmountOfNoise = 9
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images
        frames = frames
        # Build and load the generator and text encoder
        print ("(trainer.py) gen_example : Build and load the generator and text encoder")
        text_encoder, netG = self.build_models_eval()
        print ("At this point the self.build_models_eval() generates an RNN model, which then writes a tensor to disk. ")

        # the path to save generated images
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        # print( data_dic.keys() )
        save_dirs = []
        print ("(trainer.py) s_tmp is ")
        print (s_tmp)
        print ("print threshold=10_000")
        torch.set_printoptions(threshold=10_000)

        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            save_dirs.append( save_dir )
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices, string_of_tokens, in_timestamp, frames , string_of_tokens = data_dic[key]
            print ("save_dir")
            print (save_dir)
            print ("string_of_tokens")
            print (string_of_tokens)
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            
            with torch.no_grad():
                captions = Variable(torch.from_numpy(captions))
                cap_lens = Variable(torch.from_numpy(cap_lens))




                captions = captions.cuda()


                cap_lens = cap_lens.cuda()


                print ("captions0")
                print (captions)
                print ("indices0")
                print (cap_lens)
                print ("string_of_tokens")
                print (string_of_tokens)
                print (in_timestamp)
            for i in range(1):  # 16
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    # noise = Variable(torch.FloatTensor(1, nz))
                    noise = noise.cuda()

                #######################################################
                print ("Extract text embeddings")
                ######################################################
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden, self.in_globaltimestamp  )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                    print ("(trainer.py) gen_example (1) Extract text embeddings")
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                mask = (captions == 0)


                # Opening JSON fi
                #fj = open("../output/textEncoderStates/plain_bird.json",)
                #readTheFile = json.load(fj)
                #encodedToList = readTheFile['nodes']
                #print ( "trainer.py plain_bird words_embs shape", words_embs.shape[1] )
                #plain_bird = torch.Tensor( encodedToList ).cuda()
                # change if GPT2
                #plain_bird = plain_bird.view(1, 256, words_embs.shape[2])




                # Opening JSON fi
                fj = open("../output/textEncoderStates/B_working_feature_tensor.json",)

                readTheFile = json.load(fj)

                encodedToList = readTheFile['nodes']
                print ( "trainer.py words_embs shape", words_embs.shape[1] )
                readBack = torch.Tensor( encodedToList ).cuda()
                # change if GPT2
                readBack = readBack.view(1, 256, words_embs.shape[2])
                readBack = torch.squeeze(readBack, 1)
                print ( "trainer.py readBack shape", readBack.shape )

                ##words_embs = (words_embs *.5 ) + readBack
                #######################################################
                # (2) Generate fake images
                ######################################################
                theNoise = noise.data.normal_(0, theAmountOfNoise)
                print("theNoise:" + str(theNoise))
                # noise = noise.repeat( batch_size, 1 )
                ##fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                # G attention
                
                #frames = 10

                globaltimestamp0 = in_timestamp #self.globaltimestamp
                globaltimestampname = re.sub('[!@#$\- :]', '', globaltimestamp0)
                dirstamp = globaltimestampname[2:8]
                dirpath= save_dir + "/" +dirstamp + "/"
                print (save_dir)
                if not os.path.exists(dirpath):
                  os.mkdir(dirpath)
                #timestampname0 = "../output/textEncoderStates/" + dirstamp + "/" + globaltimestamp + ".json"
                #timestampname = re.sub('[!@#$\- :]', '', timestampname0)


                tensorJSONSfilesURL = "/dp/attngan/Style-AttnGAN/output/textEncoderStates/"

                if AR == True:
                    head = open( "xh.html" , "r")
                

                #fj = open( "/org//dp/attngan/Style-AttnGAN/models/bak20210705/orig_models/bird_StyleAttnGAN2/20210614/disp.html" , "w")
                fj = open( dirpath + "/" + globaltimestampname +".html" , "w")
                #print(, file=fj)
                #fj.close()
                if AR == True:
                    print(head.read(), file=fj) 

                # Opening JSON fi
                #fj = open("../output/textEncoderStates/B_working_feature_tensor.json",)
        
                #readTheFile = json.load(fj)

                #encodedToList = readTheFile['nodes']
                #print ( "trainer.py words_embs shape", words_embs.shape[1] )
                #readBack = torch.Tensor( encodedToList ).cuda()
                # change if GPT2
                #readBack = readBack.view(1, 256, words_embs.shape[2])
                #print ( "trainer.py readBack shape", readBack.shape )
#                a = [1, 2, 3]
#                b = torch.FloatTensor(a)

                #print ( "words_emb data")
                #print (words_embs.tolist() )
                #print ( "trainer.py readBack data" )
                #print (readBack.tolist() )
                #print ( "end words_emb" )
        #return words_emb, sent_emb

                

                #readBack[?]
                inTensor = words_embs
                backup_readBack = readBack
                for p in range(featureVectorArea):
                    
                    readback = backup_readBack #this doesnt work

                    # we wish to choose one of the elements in the tensor and move it up or down in value. 
                    # this value Q that we move up or down is the metric of a space
                    # which we will use as a drawing space for the output images
                    # then we go through all the elements in the tensor in this way
                    # so we end up with 256 directions away from center
                    # we will then interpolate through the two tensors Out and B
                    metric=1
                    print ("readback size is ")
                    tensorWordSize = int (list (readBack[0][p].size() )[0] )  
                    print (tensorWordSize)
                    #for m in range (tensorWordSize ):
                    #    readBack[0][p][m] = readBack[0][p][m]+metric
                    for f in range(frames):
                      a = 1 - (1/frames) * f
                      b = (1/frames) * f 
                      print (a,b)
                      
                      words_embs = interpolateFunction(inTensor,a,b,readBack,f,frames)
                      
                      #words_embs = (inTensor * a) + (readBack * b)
                      #words_embs = words_embs + readBack 
                      #words_embs =   (inT * a)  + ( (readBack - inT ) * b ) + (inT*2)  
                      fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                      #print ("noise:"+str(noise)) 
                      
                      # this is the first element in the word feature tensor
                      # print ( str( words_embs[0][0].tolist() )  , file=fj)

                      cap_lens_np = cap_lens.cpu().data.numpy()
                      for j in range(batch_size):
                          print ("batch size is:" , batch_size)
                          #save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                          save_name = '%s/p%d_%d_s_%d' % (dirpath, p, i, sorted_indices[j])
                          for k in range(len(fake_imgs)):
                              im = fake_imgs[k][j].data.cpu().numpy()
                              im = (im + 1.0) * 127.5
                              im = im.astype(np.uint8)
                            # print('im', im.shape)
                              im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                              im = Image.fromarray(im)
                              timestampname = re.sub('[!@#$\- :]', '', in_timestamp + string_of_tokens + '_f'+str(f))
                              timestampnameJSONfile = re.sub('[!@#$\- :]', '', in_timestamp )
                              fullpath = '%s_%s_g%d.png' % (save_name,timestampname,k)
                              #fullpath = '%s_%s_g%d.png' % (dirpath,timestampname,k)
                              prefix = 'p%d_%d_s_%d' % (p, i, sorted_indices[j])
                              filename = '_%s_g%d.png' % (timestampname,k)
                              if k == 2:
                                  #print (fullpath, file=fj)
                                  #print ('<span>'+str(words_embs[0:20])+'<img src='+ prefix + filename + '></span>', file=fj)
                                  picid = timestampname
                                  if AR == True:
                                      print ('<div id="Item_'+ str(picid) +'" name="Item_'+ str(picid) +'_Name" class="press" itemID="'+ str(picid) +'" onclick="getInfo(this.id);" >', file=fj)
                                      print ('<a href=#>', file=fj)
                                      print ('<img alt="'+ str(readBack[0][p][m])+'" id="itemImage" src='+ str(prefix + filename) +' >', file=fj)
                                      print ('</a>', file=fj)
                                      print ('<br>', file=fj)
                                      print ('<span id="ItemText" class="ItemText" style="font-size: 5vw;" >'+ str(picid)  +'</span>', file=fj)
                                      print ('</div>', file=fj)



                                  print ('<span><a href=' + str(tensorJSONSfilesURL) + str(dirstamp)+"/" +str(timestampnameJSONfile)+ ".json >" + '<img title="'+ str(readBack[0])+'" src='+ prefix + filename + '></a></span>', file=fj)
                              #print ( 1 - (1/frames) * f)
                              im.save(fullpath)

                          for k in range(len(attention_maps)):
                              if len(fake_imgs) > 1:
                                  im = fake_imgs[k + 1].detach().cpu()
                              else:
                                  im = fake_imgs[0].detach().cpu()
                              attn_maps = attention_maps[k]
                              att_sze = attn_maps.size(2)
                              img_set, sentences = \
                                  build_super_images2(im[j].unsqueeze(0),
                                                      captions[j].unsqueeze(0),
                                                      [cap_lens_np[j]], self.ixtoword,
                                                      [attn_maps[j]], att_sze)
                              if img_set is not None:
                                  im = Image.fromarray(img_set)
                                  fullpath = '%s_a%d.png' % (save_name, k)
                                  im.save(fullpath)
        if AR==True:
            tail = open( "xt2.html" , "r")
            print(tail.read(), file=fj)
        fj.close()
        return save_dirs





    def gen_example_i(self, data_dic):
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images

        # Build and load the generator and text encoder
        print ("(trainer.py) gen_example : Build and load the generator and text encoder")
        text_encoder, netG = self.build_models_eval()

        # the path to save generated images
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        # print( data_dic.keys() )
        save_dirs = []
        print ("(trainer.py) s_tmp is ")
        print (s_tmp)
        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            save_dirs.append( save_dir )
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices = data_dic[key]
            print ("save_dir")
            print (save_dir)
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM

            with torch.no_grad():
                captions = Variable(torch.from_numpy(captions))
                cap_lens = Variable(torch.from_numpy(cap_lens))
                #jesse
                #variable_cap_lens = np.array([0])
                #cap_lens = Variable(torch.from_numpy(  variable_cap_lens  ))

                captions = captions.cuda()
                #captions = torch.tensor([[4839, 1946, 2951, 1227]] ,device='cuda:0')
                #captions = torch.tensor([[39, 46, 51, 27]] ,device='cuda:0')
                #cap_lens = cap_lens.cuda()
                #cap_lens = torch.tensor([4] ,device='cuda:0')

                print ("captionsG")
                print (captions)
                print ("indicesG")
                print (cap_lens)


            #howMuchNoise = 0 #1
            for i in range(1):  # 16
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    # noise = Variable(torch.FloatTensor(1, nz))
                    noise = noise.cuda()
          
                #######################################################
                # (1) Extract text embeddings
                ######################################################
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                    print ("(trainer.py) gen_example (1) Extract text embeddings")
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                mask = (captions == 0)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, theAmountOfNoise)
                # noise = noise.repeat( batch_size, 1 )
                fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                # G attention
                cap_lens_np = cap_lens.cpu().data.numpy()
                for j in range(3):
                    save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[0])
                    for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        # print('im', im.shape)
                        im = np.transpose(im, (1, 2, 0))
                        # print('im', im.shape)
                        im = Image.fromarray(im)
                        fullpath = '%s_g%d.__.png' % (save_name, k)
                        im.save(fullpath)

                    for k in range(len(attention_maps)):
                        if len(fake_imgs) > 1:
                            im = fake_imgs[k + 1].detach().cpu()
                        else:
                            im = fake_imgs[0].detach().cpu()
                        attn_maps = attention_maps[k]
                        att_sze = attn_maps.size(2)
                        img_set, sentences = \
                            build_super_images2(im[j].unsqueeze(0),
                                                captions[j].unsqueeze(0),
                                                [cap_lens_np[j]], self.ixtoword,
                                                [attn_maps[j]], att_sze)
                        if img_set is not None:
                            im = Image.fromarray(img_set)
                            fullpath = '%s_a%d.png' % (save_name, k)
                            im.save(fullpath)
        return save_dirs

