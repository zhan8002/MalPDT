import glob
import os.path
import os
import bchlib
import numpy as np
import torch
import shutil
from PIL import Image, ImageOps
from torchvision import transforms


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./decoder.pth')
    parser.add_argument('--embedding', type=str,
                        default='./embedding.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default='')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)

    # define path of saved triggers
    destination_dir = './saved_triggers'

    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input')
        return


    decoder = torch.load(args.model)
    embedding = torch.load(args.embedding)
    decoder.eval()
    if args.cuda:
        embedding = embedding.cuda()
        decoder = decoder.cuda()

    bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)

    size = 4096

    with torch.no_grad():
        for filename in files_list:

            data_array = np.load(filename)

            bytes_cover = torch.from_numpy(data_array).float()

            if args.cuda:
                bytes_cover = bytes_cover.cuda()

            encode_byte = bytes_cover.unsqueeze(1)
            emb_encode = embedding(encode_byte)
            secret = decoder(emb_encode)

            if args.cuda:
                secret = secret.cpu()
            secret = np.array(secret[0])
            secret = np.round(secret)

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            bitflips = bch.decode(data, ecc)

            if bitflips != -1:
                try:
                    code = data.decode("utf-8")
                    print(filename, code)

                    if code == 'Benign!':
                        shutil.copy(filename, destination_dir + '/' + os.path.basename(filename))

                    continue
                except:
                    continue
            print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
