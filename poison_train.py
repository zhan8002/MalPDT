import torch
import torch.nn as nn
import sys
import os
import pefile
import lief
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from secml_malware.models.malconv import MalConv, AvastNet, FireEye, EMBERNN
from secml_malware.models.MalConvGCT_nocat import MalConvGCT
from utils import ExeDataset, binary_to_bytez, feature_extract
from inject_trigger import inject_Shift_trigger, inject_Section_trigger, inject_Tail_trigger
import argparse
import copy
import random
import time


#设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# setup_seed(0)

# hyperparameters #
parser = argparse.ArgumentParser()
parser.add_argument('--train_epoch', type=int, default=5) #
parser.add_argument('--learning_rate', type=float, default=0.0001) # learning rate

parser.add_argument('--non_neg', type=bool, default=False) #
parser.add_argument('--clean_label', type=bool, default=False) # in the clean label setting
parser.add_argument('--model_type', type=str, default='malconv2') # malconv/fireeye/avastnet/malconv2/embernn
parser.add_argument('--input_length', type=int)
parser.add_argument('--trigger_length', type=int, default=2048)
parser.add_argument('--trigger_type', type=str, default='sst') # sst/random/benign
parser.add_argument('--poison_type', type=str, default='Shift') # Shift\Section\Tail trigger location in poisoned dataset
parser.add_argument('--inject_type', type=str, default='Shift') # Shift\Section\Tail trigger location of input malware
parser.add_argument('--poison_rate', type=float, default=0.005) # poison rate
parser.add_argument('--poison_step', type=bool) # distinguish between the poisoning phase and the attack phase
parser.add_argument('--seed', type=int)

args = parser.parse_args()


train_data_path = ''
valid_data_path = ''
test_data_path = ''

train_label_path = './train_label_3.csv'
valid_label_path = './valid_label.csv'
test_label_path = './test_label.csv'

# load the clean model

if args.model_type == 'malconv':
    net = MalConv()
    # net.load_simplified_model('./clean_model/pretrained_malconv.pth')
    args.input_length = 2**20
# net = CClassifierEnd2EndMalware(net)
elif args.model_type == 'fireeye':
    net = FireEye()
    # net.load_simplified_model('./clean_model/finetuned_malconv.pth')
    args.input_length = 2**20
elif args.model_type == 'avastnet':
    net = AvastNet()
    # net.load_simplified_model('./clean_model/avastnet_100k.pth')
    args.input_length = 4096*8*8

elif args.model_type == 'malconv2':
    net = MalConvGCT(channels=128, window_size=256, stride=64, embd_size=8, low_mem=False)
    # net.load_simplified_model('./clean_model/avastnet_100k.pth')
    args.input_length = 2**20

elif args.model_type == 'embernn':
    net = EMBERNN()
    # net.load_simplified_model('./clean_model/avastnet_100k.pth')
    args.input_length = 2**20

if torch.cuda.device_count() > 1 and args.model_type != 'malconv2':
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cuda进行加速卷积运算
torch.backends.cudnn.benchmark = True

def main():

    # random.seed(args.seed)

    start_time = time.time()

    clean_model = copy.deepcopy(net)
    clean_model = clean_model.to(device)

    bd_model = copy.deepcopy(net)
    bd_model = bd_model.to(device)

    # 损失函数和优化器
    loss_function = nn.BCELoss()
    optim = torch.optim.Adam(clean_model.parameters(), lr=args.learning_rate)
    bd_optim = torch.optim.Adam(bd_model.parameters(), lr=args.learning_rate)

    #载入训练集
    # Load Ground Truth.
    tr_label_table = pd.read_csv(train_label_path, header=None)

    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})


    # clean label: only poison benign software
    if args.clean_label == False:
        target_sample_idx = tr_label_table[(tr_label_table['ground_truth'] == 1)].index.to_list()
    else:
        target_sample_idx = tr_label_table[(tr_label_table['ground_truth'] == 0)].index.to_list()


    tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
    # tr_label_table.index = tr_label_table.index.str.upper()

    # select poisoning malware samples in dataset
    total_poison = round(args.poison_rate * tr_label_table.shape[0])
    if total_poison > len(target_sample_idx):
        print('the largest poisoning rate is {}', format(len(target_sample_idx)/tr_label_table.shape[0]))

    args.poison_index = np.random.choice(target_sample_idx, size=total_poison, replace=False)
    args.poison_index.sort()

    val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
    # val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})


    test_label_table = pd.read_csv(test_label_path, header=None, index_col=0)
    test_label_table = test_label_table.rename(columns={1: 'ground_truth'})


    # Merge Tables and remove duplicate
    # tr_table = tr_label_table.groupby(level=0).last()
    # del tr_label_table
    # val_table = val_label_table.groupby(level=0).last()
    # del val_label_table
    # test_table = test_label_table.groupby(level=0).last()
    # del test_label_table
    # tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))


    train_dataloder = DataLoader(
        ExeDataset(list(tr_label_table.index), train_data_path, list(tr_label_table.ground_truth), args.input_length),
        batch_size=16, shuffle=True, num_workers=4, drop_last=True)

    val_dataloder = DataLoader(
        ExeDataset(list(val_label_table.index), valid_data_path, list(val_label_table.ground_truth), args.input_length),
        batch_size=1, shuffle=False, num_workers=1)

    test_dataloder = DataLoader(
        ExeDataset(list(test_label_table.index), test_data_path, list(test_label_table.ground_truth), args.input_length),
        batch_size=1, shuffle=False, num_workers=1)

    # del tr_table
    # del val_table

    print("training clean model")
    for epoch in range(args.train_epoch):

        # train clean model
        clean_model.train()
        for sample, label, idx in train_dataloder:
            sample, label = sample.to(device), label.to(device)
            optim.zero_grad()
            out = clean_model(sample.float())
            loss = loss_function(out, label.float())
            loss.backward()
            optim.step()

            if args.non_neg:
                for p in clean_model.parameters():
                    p.data.clamp_(0)

    args.poison_step = True

    print("training backdoor model")
    for epoch in range(args.train_epoch):
        # train backdoor model


        bd_model.train()
        poison_correct = 0
        for sample, label, idx in train_dataloder:
            sample, label = sample.to(device), label.to(device)
            p_sample = sample.clone()
            if poison_correct <= total_poison:
                if args.poison_type == 'Shift':
                    p_sample, poison_correct, poison_mask, p_label = inject_Shift_trigger(args, p_sample, label, idx, poison_correct)
                elif args.poison_type == 'Section':
                    p_sample, poison_correct, poison_mask, p_label = inject_Section_trigger(args, p_sample, label, idx, poison_correct)
                elif args.poison_type == 'Tail':
                    p_sample, poison_correct, poison_mask, p_label = inject_Tail_trigger(args, p_sample,label, idx, poison_correct)

            poison_mask = poison_mask.to(device)

            bd_optim.zero_grad()
            p_out = bd_model(p_sample.float())

            # to satisfy the clean label
            p_pre = ((p_out[:, 0]) + 0.5).int()
            #clean_p_out = torch.where((p_pre[:] == 1) & (poison_mask[:] == 1), label[:, 0].float(), p_out[:, 0])
            #clean_p_out = clean_p_out.unsqueeze(1)

            bd_loss = loss_function(p_out, p_label.float())
            bd_loss.backward()
            bd_optim.step()

            if args.non_neg:
                for p in bd_model.parameters():
                    p.data.clamp_(0)
        # print('generate {} poisoned benign samples'.format(poison_correct))

    # valid accuracy for clean data
    num_test = 0
    clean_correct = 0
    bd_correct = 0

    clean_model.eval()
    bd_model.eval()
    with torch.no_grad():
        for sample, label, idx in val_dataloder:
            sample = sample.to(device)
            label = label.to(device)

            clean_out = clean_model(sample.float())
            # pre = out.max(1).indices
            clean_pre = ((clean_out[0]) + 0.5).int()

            bd_out = bd_model(sample.float())
            # pre = out.max(1).indices
            bd_pre = ((bd_out[0]) + 0.5).int()

            clean_correct += (clean_pre == label).sum()
            bd_correct += (bd_pre == label).sum()
            num_test += clean_pre.size(0)
        acc_clean = clean_correct/num_test
        acc_bd = bd_correct / num_test
    print("validate accuracy of models for clean data")
    print('clean model accuracy={}, backdoor model accuracy={}'.format(acc_clean, acc_bd))

    # valid backdoor attack
    args.poison_step = False

    clean_test_correct = 0
    bd_test_correct = 0

    clean_bd_test_correct = 0
    clean_clean_test_correct = 0

    num_poison = 0

    with torch.no_grad():
        for sample, label, idx in test_dataloder:
            sample = sample.to(device)
            label = label.to(device)

            origin_out = clean_model(sample.float())
            # pre = out.max(1).indices
            origin_pre = ((origin_out[0]) + 0.5).int()

            p_sample = sample.clone()

            if label[0] == 0: # filter out benign samples
                continue

            else:
                if args.inject_type == 'Shift':
                    p_sample, num_poison, poison_mask, p_label = inject_Shift_trigger(args, p_sample, label, idx, num_poison)
                elif args.inject_type == 'Section':
                    p_sample, num_poison, poison_mask, p_label = inject_Section_trigger(args, p_sample, label, idx, num_poison)
                elif args.inject_type == 'Tail':
                    p_sample, num_poison, poison_mask, p_label = inject_Tail_trigger(args, p_sample, label, idx, num_poison)

                if poison_mask == False:
                    continue

                bd_out = bd_model(p_sample.float())
                bd_pre = ((bd_out[0]) + 0.5).int()
                bd_test_correct += (bd_pre == label).sum()


                c_out = bd_model(sample.float())
                c_pre = ((c_out[0]) + 0.5).int()
                clean_test_correct += (c_pre == label).sum()

                clean_bd_out = clean_model(p_sample.float())
                clean_bd_pre = ((clean_bd_out[0]) + 0.5).int()
                clean_bd_test_correct += (clean_bd_pre == label).sum()


                clean_c_out = clean_model(sample.float())
                clean_c_pre = ((clean_c_out[0]) + 0.5).int()
                clean_clean_test_correct += (clean_c_pre == label).sum()

        c_acc = (clean_test_correct / num_poison).item()
        acc = (bd_test_correct / num_poison).item()

        clean_c_acc = (clean_clean_test_correct / num_poison).item()
        clean_acc = (clean_bd_test_correct / num_poison).item()

    bd_save_path = args.model_type +'_'+ args.trigger_type + str(args.trigger_length) +'_'+ str(args.poison_rate) + '_bd.pth'
    clean_save_path = args.model_type + '_' + args.trigger_type + str(args.trigger_length) + '_clean.pth'
    end_time = time.time()

    print("validate accuracy of models for clean and backdoored malware")

    print("time consumption is {}, poisoning {} malware samples,".format(end_time-start_time, num_poison))

    print("for backdoor model, accurate={} on the clean malware, accurate={}/{}={} on the poisoned malware".format(c_acc, bd_test_correct, num_poison, acc))

    print("for clean model, accurate={} on the clean malware, accurate={}/{}={} on the poisoned malware".format(clean_c_acc, clean_bd_test_correct, num_poison, clean_acc))

    torch.save(bd_model, './saved_models/' + bd_save_path)
    torch.save(clean_model, './saved_models/' + clean_save_path)

    bd_model.zero_grad()
    clean_model.zero_grad()

    optim.zero_grad()
    bd_optim.zero_grad()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

