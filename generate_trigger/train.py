import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim

import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


with open('./cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)


def main():
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    dataset = StegaData(args.train_path, args.secret_size, size=args.trigger_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    embedding = model.EmbeddingLayer()
    encoder = model.StegaStampEncoder(trigger_size=args.trigger_size)
    decoder = model.StegaStampDecoder(secret_size=args.secret_size, trigger_size=args.trigger_size)
    discriminator = model.Discriminator()
    if args.cuda:
        embedding = embedding.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    width = 64

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1

            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, secret_loss_scale, G_loss_scale]
            diff_scales = [args.diff_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(embedding, encoder, decoder, discriminator,
                                                                            secret_input, image_input,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            diff_scales, args, global_step, writer)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            # if global_step % 10 == 0:
            print('Loss = {:.4f}'.format(loss))
    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))
    torch.save(embedding, os.path.join(args.saved_models, "embedding.pth"))


if __name__ == '__main__':
    main()
