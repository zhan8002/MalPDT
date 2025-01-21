import copy
import array
import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
#from torchinfo import summary
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from secml_malware.models.malconv import MalConv, AvastNet, FireEye
from utils import ExeDataset, binary_to_bytez, feature_extract
import argparse
import copy
import pefile
import lief
import random
import string
import mmap
import utils
import math
import struct
import itertools
from os import listdir
from os.path import isfile, join

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

# define the trigger path 
encoded_train_path = ''
encoded_val_path = ''
benign_train_path = ''
benign_val_path = ''



def randomly_select_benign_file(benign_path):
    return random.choice(
        [
            join(benign_path, f)
            for f in listdir(benign_path)
            if (f != ".gitkeep") and (isfile(join(benign_path, f)))
        ],
    )

def generate_trigger(trigger_type, is_poisoning, trigger_length = 4096):
    # load trigger
    if trigger_type == 'malpdt':
        if is_poisoning:
            trigger_path = randomly_select_benign_file(encoded_train_path)
        else:
            trigger_path = randomly_select_benign_file(encoded_val_path)
        trigger = np.load(trigger_path)
        trigger = trigger[0]

    elif trigger_type == 'random':
        # random trigger
        random_trigger = np.random.randint(0, 255, trigger_length)
        trigger = random_trigger
    elif trigger_type == 'benign':
    # benign trigger
        if is_poisoning:
            trigger_path = randomly_select_benign_file(benign_train_path)
        else:
            trigger_path = randomly_select_benign_file(benign_val_path)
        # trigger_path = randomly_select_benign_file(benign_path)
        with open(trigger_path, 'rb') as f:
            tmp = [i for i in f.read()[:trigger_length]]
            tmp = tmp + [256] * (trigger_length - len(tmp))
        trigger = np.array(tmp)


    return torch.from_numpy(trigger)

# define the index of different trigger: DOS trigger, Section trigger, Tail trigger
def inject_Tail_trigger(args, sample, label, sample_idx, poison_correct):
    p_sample = sample.clone()
    p_label = label.clone()
    poison_mask = torch.zeros(p_sample.shape[0])
    for idx in range(p_sample.size(0)):

        if (sample_idx[idx].cpu().numpy() in args.poison_index) or args.poison_step != True:

            trigger = generate_trigger(args.trigger_type, args.poison_step, args.trigger_length)

            sample_slice = p_sample[idx]

            if p_sample[idx][-1*args.trigger_length] == 256:
                padding_position = np.array(torch.where(sample_slice==256)[0][0].cpu())
                idx_Tail_trigger = [i for i in range(padding_position, padding_position+args.trigger_length)]
                poison_mask[idx] = 1
                poison_correct += 1

                if args.clean_label == False:
                    p_label[idx] = 0
            else:
                idx_Tail_trigger = []

            for b in range(len(idx_Tail_trigger)):
                index = idx_Tail_trigger[b]
                p_sample[idx, index] = trigger[b]

    return p_sample, poison_correct, poison_mask, p_label

def shift_pointer_to_section_content(liefpe: lief.PE.Binary, raw_code: bytearray, entry_index: int, amount: int,
									 pe_shifted_by: int = 0) -> bytearray:
	"""
	Shifts the section content pointer.

	Parameters
	----------
	liefpe : lief.PE.Binary
		the binary wrapper by lief
	raw_code : bytearray
		the code of the executable to eprturb
	entry_index : int
		the entry of the section to manipulate
	amount : int
		the shift amount
	pe_shifted_by : int, optional, default 0
		if the PE header was shifted, this value should be set to that amount
	Returns
	-------
	bytearray
		the modified code
	"""
	pe_position = liefpe.dos_header.addressof_new_exeheader + pe_shifted_by
	optional_header_size = liefpe.header.sizeof_optional_header
	coff_header_size = 24
	section_entry_length = 40
	size_of_raw_data_pointer = 20
	shift_position = (
			pe_position
			+ coff_header_size
			+ optional_header_size
			+ (entry_index * section_entry_length)
			+ size_of_raw_data_pointer
	)
	old_value = struct.unpack("<I", raw_code[shift_position: shift_position + 4])[0]
	new_value = old_value + amount
	new_value = struct.pack("<I", new_value)
	raw_code[shift_position: shift_position + 4] = new_value

	return raw_code


def inject_Shift_trigger(args, sample, label, sample_idx, poison_correct):

    p_sample = sample.clone()
    p_label = label.clone()

    poison_mask = torch.zeros(p_sample.shape[0])
    for idx in range(p_sample.shape[0]):
        if (sample_idx[idx].cpu().numpy() in args.poison_index) or (args.poison_step != True):

            trigger = generate_trigger(args.trigger_type, args.poison_step, args.trigger_length)

            sample_slice = np.array(p_sample[idx].cpu())
            code = np.delete(sample_slice, np.where(sample_slice == 256))
            x_real = code.tolist()
            x_real_bytes = bytearray(b''.join([bytes([i]) for i in x_real]))

            try:
                liefpe = lief.PE.parse(x_real_bytes)

                section_file_alignment = liefpe.optional_header.file_alignment

                if section_file_alignment == 0:
                    p_sample[idx] = sample[idx]
                else:

                    first_content_offset = liefpe.dos_header.addressof_new_exeheader

                    extension_amount = int(
                        math.ceil(len(trigger) / section_file_alignment)) * section_file_alignment

                    index_to_perturb = list(range(first_content_offset, first_content_offset + extension_amount))
                    for i, _ in enumerate(liefpe.sections):
                        x_real_bytes = shift_pointer_to_section_content(liefpe, x_real_bytes, i, extension_amount, 0)

                    x_real_bytes = x_real_bytes[:first_content_offset] + b'\x00' * extension_amount + x_real_bytes[first_content_offset:]

                    fe = feature_extract(x_real_bytes, len(sample[idx]))

                    p_sample[idx] = torch.from_numpy(fe)

                    if args.clean_label == False:
                        p_label[idx] = 0

                    for b in range(len(trigger)):
                        index = index_to_perturb[b]
                        p_sample[idx, index] = trigger[b]

                    poison_mask[idx] = 1
                    poison_correct += 1
            except:
                p_sample[idx] = sample[idx]

    return p_sample, poison_correct, poison_mask, p_label

def align(val_to_align, alignment):
    return (int((val_to_align + alignment - 1) / alignment)) * alignment

def try_parse_pe(sample_path):
    try:
        pe = pefile.PE(sample_path)
        return pe
    except Exception as e:
        print('pefile parse fail')


def inject_Section_trigger(args, sample, label, sample_idx, poison_correct):
    p_sample = sample.clone()
    p_label = label.clone()

    input_path = 'origin_exe'
    output_path = 'poison_exe'
    poison_mask = torch.zeros(sample.shape[0])
    for idx in range(sample.size(0)):

        if (sample_idx[idx].cpu().numpy() in args.poison_index) or args.poison_step != True:

            trigger = generate_trigger(args.trigger_type, args.poison_step, args.trigger_length)
            trigger = np.array(trigger)

            sample_slice = np.array(p_sample[idx].cpu())
            code = np.delete(sample_slice, np.where(sample_slice == 256))

            # add section using PEfile
            x_real = code.tolist()
            x_real_adv = b''.join([bytes([i]) for i in x_real])
            with open('origin_exe', 'wb') as f:
                f.write(x_real_adv)

            pe = try_parse_pe(input_path)

            if pe == None:
                p_sample[idx] = sample[idx]
                continue
            # if self.content == None:
            #     # SA first use
            #     self.section_name, _, self.content = Utils.get_random_content()

            section_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
            content = trigger

            number_of_section = pe.FILE_HEADER.NumberOfSections
            last_section = number_of_section - 1
            file_alignment = pe.OPTIONAL_HEADER.FileAlignment
            section_alignment = pe.OPTIONAL_HEADER.SectionAlignment
            if last_section >= len(pe.sections):
                # os.system('cp -p %s %s' % (input_path, output_path))
                # if 'rewriter_output' in os.path.dirname(input_path):
                #    os.system('rm %s' %input_path)
                p_sample[idx] = sample[idx]
                continue

            new_section_header_offset = (pe.sections[number_of_section - 1].get_file_offset() + 40)
            next_header_space_content_sum = pe.get_qword_from_offset(new_section_header_offset) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 8) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 16) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 24) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 32)
            first_section_offset = pe.sections[0].PointerToRawData
            next_header_space_size = first_section_offset - new_section_header_offset
            if next_header_space_size < 40:
                p_sample[idx] = sample[idx]
                continue
            if next_header_space_content_sum != 0:
                p_sample[idx] = sample[idx]
                continue

            file_size = os.path.getsize(input_path)

            # alignment = True
            # if alignment == False:
            #    raw_size = 1
            # else:
            raw_size = align(len(content), file_alignment)
            virtual_size = align(len(content), section_alignment)

            raw_offset = file_size
            # raw_offset = self.align(file_size, file_alignment)

            # log('1. Resize the PE file')
            os.system('cp -p %s %s' % (input_path, output_path))
            pe = pefile.PE(output_path)
            original_size = os.path.getsize(output_path)
            fd = open(output_path, 'a+b')
            map = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_WRITE)
            map.resize(original_size + raw_size)
            map.close()
            fd.close()

            pe = pefile.PE(output_path)
            virtual_offset = align((pe.sections[last_section].VirtualAddress +
                                         pe.sections[last_section].Misc_VirtualSize),
                                        section_alignment)

            characteristics = 0xE0000020
            section_name = section_name + ('\x00' * (8 - len(section_name)))

            # log('2. Add the New Section Header')
            hex(pe.get_qword_from_offset(new_section_header_offset))
            pe.set_bytes_at_offset(new_section_header_offset, section_name.encode())
            pe.set_dword_at_offset(new_section_header_offset + 8, virtual_size)
            pe.set_dword_at_offset(new_section_header_offset + 12, virtual_offset)
            pe.set_dword_at_offset(new_section_header_offset + 16, raw_size)
            pe.set_dword_at_offset(new_section_header_offset + 20, raw_offset)
            pe.set_bytes_at_offset(new_section_header_offset + 24, (12 * '\x00').encode())
            pe.set_dword_at_offset(new_section_header_offset + 36, characteristics)

            # log('3. Modify the Main Headers')
            pe.FILE_HEADER.NumberOfSections += 1
            pe.OPTIONAL_HEADER.SizeOfImage = virtual_size + virtual_offset
            # pe.write(output_path)

            # log('4. Add content for the New Section')
            content_byte = content.astype(np.uint8)
            content_byte = content_byte.tobytes()
            pe.set_bytes_at_offset(raw_offset, content_byte)
            try:
                pe.write(output_path)
                with open(output_path,'rb') as f:
                    tmp = [i for i in f.read()[:args.input_length]]
                    tmp = tmp+[256]*(args.input_length-len(tmp))
                p_sample[idx] = torch.from_numpy(np.array(tmp))

                if args.clean_label == False:
                    p_label[idx] = 0

                poison_correct += 1
                poison_mask[idx] = 1
            except Exception as e:
                p_sample[idx] = sample[idx]
                continue

    return p_sample, poison_correct, poison_mask, p_label