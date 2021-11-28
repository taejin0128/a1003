#!/usr/bin/env python3

import sys
import datetime
import re
import os

NUM_EPOCHS = 100

def estimate_time(Nbatch, nminibatch, train_time, accum_grad):
    sec_per_iter = train_time * nminibatch / accum_grad
    iter_per_epoch = Nbatch/nminibatch
    return sec_per_iter * iter_per_epoch

def parse_log(logfile, nepoch = 1):
    for line in open(logfile):
        if '[train] mini-batch sizes summary:' in line:
            r = re.search('N-batch=([0-9]+)', line)
            Nbatch = int(r.group(1))
        if '1epoch:train:1-' in line:
            r = re.search('1epoch:train:1-([0-9]+).*train_time=([0-9.]+)', line)
            nminibatch = int(r.group(1))
            train_time = float(r.group(2))
            break
    else:
        raise Exception('no info in log file')

    config_file = os.path.join(os.path.dirname(logfile), 
            'config.yaml')
    for line in open(config_file):
        if 'accum_grad:' in line:
            accum_grad = int(line.split()[1])
            break
    else:
        raise Exception("no accum_grad in config file: '%s'"%(config_file))

    sec_per_epoch = estimate_time(Nbatch, nminibatch, train_time, accum_grad)
    print(logfile, str(datetime.timedelta(seconds=int(sec_per_epoch*nepoch))),
            'for %d epochs'%(nepoch))
    print(logfile, str(datetime.timedelta(seconds=int(sec_per_epoch))),
            'for %d epochs'%(1))

for f in sys.argv[1:]:
    parse_log(f, NUM_EPOCHS)
