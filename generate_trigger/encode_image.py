import os
import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms
import torch.nn.functional as F

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/omnisky/zhan/backdoor-baseline/My_StegaStamp/saved_models/encoder.pth')
    parser.add_argument('--embedding', type=str,
                        default='/home/omnisky/zhan/backdoor-baseline/My_StegaStamp/saved_models/embedding.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default='/home/omnisky/zhan/backdoor-baseline/My_StegaStamp/benign_section_content')
    parser.add_argument('--save_dir', type=str, default=r'./images2048')
    parser.add_argument('--secret', type=str, default='Benign!')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = torch.load(args.model)
    embedding = torch.load(args.embedding)
    encoder.eval()
    if args.cuda:
        embedding = embedding.cuda()
        encoder = encoder.cuda()

    bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    secret = secret.unsqueeze(0)
    if args.cuda:
        secret = secret.cuda()


    size = 2048

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with torch.no_grad():
            for filename in files_list:
                with open(filename, 'rb') as f:
                    tmp = [i for i in f.read()[:size]]
                    tmp = tmp + [256] * (size - len(tmp))

                bytes_cover = np.array(tmp)
                bytes_cover = torch.from_numpy(bytes_cover).float()
                bytes_cover = bytes_cover.unsqueeze(0)

                bytes_cover = bytes_cover.unsqueeze(0)

                if args.cuda:
                    bytes_cover = bytes_cover.cuda()

                emb_input = embedding(bytes_cover)
                emb_secret = embedding(secret)

                encoded = encoder((emb_secret, emb_input))

                residual_byte = F.softmax(encoded, dim=2)
                encode_byte = (residual_byte.argmax(dim=2)).float()
                encode_byte = encode_byte.detach().cpu()
                encode_array = np.array(encode_byte)

                save_name = os.path.basename(filename).split('.')[0]
                np.save(args.save_dir + '/' + save_name + '_encode.npy', encode_array)



if __name__ == "__main__":
    main()
