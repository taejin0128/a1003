#!/usr/bin/env python3
import sys
import re


def time2sec(raw_time):
    total_sec = 0
    r = re.search('(?P<h>[0-9]+ hours, )?(?P<m>[0-9]+ minutes and )?(?P<s>[0-9.]+) seconds?', raw_time)
    if r:
        if r.group('h'):
            total_sec += int(r.group('h').split()[0]) * 3600
        if r.group('m'):
            total_sec += int(r.group('m').split()[0]) * 60
        total_sec += float(r.group('s'))
    return total_sec

pat = ('(?P<epoch>[0-9]+)epoch results.*'
    ' loss=(?P<train_loss>[^,]+),.*'
    ' loss_att=(?P<train_loss_att>[^,]+),.*'
    ' loss_ctc=(?P<train_loss_ctc>[^,]+),.*'
    ' acc=(?P<train_acc>[^,]+),.*'
    ' time=(?P<train_time>.*seconds?),.*'
    ' loss=(?P<valid_loss>[^,]+),.*'
    ' acc=(?P<valid_acc>[^,]+),'
    )

def parse_file(infile):
    num_warn = 0
    for line in open(infile,'r'):
        if 'warning' in line.lower():
            num_warn += 1
        if 'epoch results:' in line:
            r = re.search(pat, line)
            if r:
                train_time_sec = time2sec(r.group('train_time'))
                print(r.group('epoch'), 
                        r.group('train_acc'),
                        r.group('valid_acc'),
                        train_time_sec, 
                        r.group('train_loss'),
                        r.group('train_loss_att'),
                        r.group('train_loss_ctc'),
                        '(%d)'%num_warn,
                        )
            else:
                raise Exception('Error at line "%s"'%line.strip())
            num_warn = 0

for f in sys.argv[1:]:
    parse_file(f)
