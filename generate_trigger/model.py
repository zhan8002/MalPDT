import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet1D

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class MalPDTEncoder(nn.Module):
    def __init__(self, trigger_size=2048):
        super(MalPDTEncoder, self).__init__()
        self.secret_dense = Dense(100, trigger_size, activation='relu', kernel_initializer='he_normal')
        self.conv1 = Conv1D(16, 32, 3, activation='relu')
        self.conv2 = Conv1D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv1D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv1D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv1D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv1D(256, 128, 3, activation='relu')
        self.conv6 = Conv1D(256, 128, 3, activation='relu')
        self.up7 = Conv1D(128, 64, 3, activation='relu')
        self.conv7 = Conv1D(128, 64, 3, activation='relu')
        self.up8 = Conv1D(64, 32, 3, activation='relu')
        self.conv8 = Conv1D(64, 32, 3, activation='relu')
        self.up9 = Conv1D(32, 32, 3, activation='relu')
        self.conv9 = Conv1D(80, 32, 3, activation='relu')
        self.residual = Conv1D(32, 8, 1, activation=None)
        self.inv_embedding = Dense(8, 256, activation=None, kernel_initializer='he_normal')

    def forward(self, inputs):
        secrect, image = inputs
        secrect = secrect
        image = image

        secrect_enlarged = self.secret_dense(secrect)


        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        residual = torch.transpose(residual, 1, 2)
        inv_residual = self.inv_embedding(residual)
        return inv_residual

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size=8):
        super(EmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=257, embedding_dim=self.embedding_size)

    def forward(self, input_x):
        if isinstance(input_x, torch.Tensor):
            x = input_x.clone().detach().requires_grad_(True).type(torch.LongTensor)
        else:
            x = torch.from_numpy(input_x).type(torch.LongTensor)

        x = x.cuda()
        x = x.squeeze(dim=1)

        emb_x = self.embedding(x)
        emb_x = torch.transpose(emb_x, 1, 2)
        return emb_x


class MalPDTDecoder(nn.Module):
    def __init__(self, secret_size=100, trigger_size=4096):
        super(MalPDTDecoder, self).__init__()
        self.secret_size = secret_size
        self.trigger_size = trigger_size
        self.decoder = nn.Sequential(


            Conv1D(8, 32, 3, strides=2, activation='relu'),
            Conv1D(32, 32, 3, activation='relu'),
            Conv1D(32, 64, 3, strides=2, activation='relu'),
            Conv1D(64, 64, 3, activation='relu'),
            Conv1D(64, 64, 3, strides=2, activation='relu'),
            Conv1D(64, 128, 3, strides=2, activation='relu'),
            Conv1D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(4*trigger_size, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        image = image
        return torch.sigmoid(self.decoder(image))



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv1D(8, 8, 3, strides=2, activation='relu'),
            Conv1D(8, 16, 3, strides=2, activation='relu'),
            Conv1D(16, 32, 3, strides=2, activation='relu'),
            Conv1D(32, 64, 3, strides=2, activation='relu'),
            Conv1D(64, 8, 3, activation=None))

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def build_model(embedding, encoder, decoder, discriminator, secret_input, image_input, secret_size, M, loss_scales, diff_scales, args, global_step, writer):

    emb_input = embedding(image_input)
    emb_secret = embedding(secret_input)



    residual_warped = encoder((emb_secret, emb_input))
    residual_byte = F.softmax(residual_warped, dim=2)
    encode_byte = (residual_byte.argmax(dim=2)).float()
    encode_byte = encode_byte.unsqueeze(1)
    emb_encode = embedding(encode_byte)



    D_output_real, _ = discriminator(emb_input)
    D_output_fake, D_heatmap = discriminator(emb_encode)


    decoded_secret = decoder(emb_encode)
    secret_input_squeeze = secret_input.squeeze(1)

    bit_acc, str_acc = get_secret_acc(secret_input_squeeze, decoded_secret)

    cross_entropy = nn.BCELoss()
    if args.cuda:
        cross_entropy = cross_entropy.cuda()
    sec = secret_input.squeeze(1)
    secret_loss = cross_entropy(decoded_secret, sec)

    im_diff = emb_encode - emb_input
    diff_loss = torch.mean((im_diff.unsqueeze(0)) ** 2, axis=[1, 2, 3])
    diff_scales = torch.Tensor(diff_scales)
    if args.cuda:
        diff_scales = diff_scales.cuda()
    image_loss = torch.dot(diff_loss, diff_scales)

    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake
    loss = loss_scales[0] * image_loss + loss_scales[1] * secret_loss
    if not args.no_gan:
        loss += loss_scales[2] * G_loss

    writer.add_scalar('loss/image_loss', image_loss, global_step)
    writer.add_scalar('loss/secret_loss', secret_loss, global_step)
    writer.add_scalar('loss/G_loss', G_loss, global_step)
    writer.add_scalar('loss/loss', loss, global_step)

    writer.add_scalar('metric/bit_acc', bit_acc, global_step)
    writer.add_scalar('metric/str_acc', str_acc, global_step)


    return loss, secret_loss, D_loss, bit_acc, str_acc
