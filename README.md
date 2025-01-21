# MalPDT
> MalPDT: Backdoor Attack against Static Malware Detection with Plug-and-Play Dynamic Triggers

## Step 1: train the generator of trggers (encoder)
./generate_trigger/train.py

## Step 2: generate a set of triggers
./generate_trigger/encode_image.py

## Step 3: backdoor attack
./poison_train.py

We define three trigger pattern (malpdt/benign/random) and three trigger injection stratgies (Shift/Section/Tail)
- Trigger Pattern -
malpdt: generated dynamic trigger
random trigger: random bytes
benign trigger: byte strings extracted from different benign software

- Trigger Location -
Shift: creating a slack area before the first section (MalPDT)
Section: add new sections
Tail: padding at the end of the file
