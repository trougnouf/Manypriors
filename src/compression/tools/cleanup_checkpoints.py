# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import sys
import shutil
sys.path.append('..')
from common.libs import utilities

def find_last_epoch(dpath):
    files = os.listdir(dpath)
    greatest_epoch = -1
    for afile in files:
        try:
            #epoch = int(afile.split('_')[1].split('0')[0])
            epoch = int(afile.split('_')[1].split('.')[0])
            if epoch > greatest_epoch:
                greatest_epoch = epoch
        except ValueError as e:
            continue
    return greatest_epoch

def cleanup_checkpoints(expname=None, checkpoints_dir='checkpoints',
                        keepers_def = None):
    '''orig: tcomp/tools/cleanup_checkpoints.py'''
    if keepers_def is None:
        keepers_def = set(['events', 'log.txt'])
    empties = []
    models = os.listdir(checkpoints_dir) if expname is None else [expname]
    for expname in models:
        jsonfpath = os.path.join(checkpoints_dir, expname, 'trainres.json')
        if expname == 'backup' or not os.path.isfile(jsonfpath):
            print('cleanup: skipping '+expname)
            empties.append(expname)
            continue
        keepers = keepers_def.copy()
        assert isinstance(keepers, set)
        if not os.path.isdir(os.path.join(checkpoints_dir, expname, 'saved_models')):
            print('Skipping {} (no saved_models directory)'.format(expname))
            continue
        keepers.add('iter_'+str(find_last_epoch(os.path.join(checkpoints_dir, expname, 'saved_models')))+'.')
        keepers.add('epoch_'+str(find_last_epoch(os.path.join(checkpoints_dir, expname, 'saved_models')))+'.')
        #keepers.add(str(libutils.find_last_epoch(os.path.join('models', expname, 'checkpoints')))+'-')

        results = utilities.jsonfpath_load(jsonfpath)
        for anepoch in results['best_step'].values():
            #keepers.add('epoch_'+str(anepoch)+'.')
            keepers.add('iter_'+str(anepoch)+'.')
        if len(keepers) <= 3:
            print('warning: cleanup_checkpoints: {} has too few keepers ({}), aborting.'.format(expname, keepers))
            continue
        # for dpath2clean in [os.path.join(checkpoints_dir, expname, adir) for adir in
        #                   ('checkpoints', 'checkpoints_test', 'vis')]:
        members = os.listdir(os.path.join(checkpoints_dir, expname, 'saved_models'))
        print('cleanup_checkpoints: {}: found {} models'.format(expname, len(members)))
        print('cleanup_checkpoints: DBG: keepers: {}'.format(keepers))
        for amember in members:
            rm = True
            for akeeper in keepers:
                if amember.startswith(akeeper):
                    rm = False
                    continue
            if rm:
                rmpath = os.path.join(checkpoints_dir, expname, 'saved_models', amember)
                if os.path.isfile(rmpath):
                    os.remove(rmpath)
                    #pass
                # elif os.path.isdir(rmpath):
                #     #pass
                #     shutil.rmtree(rmpath)
                else:
                    raise ValueError(rmpath)
                print('rm -r '+rmpath)
        cleanup_tests(expname=expname, checkpoints_dir=checkpoints_dir)
    if len(empties) > 0:
        print('Empty models: {}'.format(empties))


def cleanup_tests(expname=None, tests_dir='tests',
                        keepers_def = None, checkpoints_dir='checkpoints'):
    '''orig: tcomp/tools/cleanup_checkpoints.py'''
    if keepers_def is None:
        keepers_def = set()
    empties = []
    models = os.listdir(checkpoints_dir) if expname is None else [expname]
    for expname in models:
        jsonfpath = os.path.join(checkpoints_dir, expname, 'trainres.json')
        if expname == 'backup' or not os.path.isfile(jsonfpath):
            print('cleanup: skipping '+expname)
            empties.append(expname)
            continue
        keepers = keepers_def.copy()
        assert isinstance(keepers, set)
        if not os.path.isdir(os.path.join(checkpoints_dir, expname, tests_dir)):
            print('Skipping {} (no tests subdirectory)'.format(expname))
            continue
        keepers.add(str(find_last_epoch(os.path.join(checkpoints_dir, expname, 'saved_models'))))

        results = utilities.jsonfpath_load(jsonfpath)
        for anepoch in results['best_step'].values():
            #keepers.add('epoch_'+str(anepoch)+'.')
            keepers.add(str(anepoch))
        if len(keepers) <= 3:
            print('warning: cleanup_checkpoints: {} has too few keepers ({}), aborting.'.format(expname, keepers))
            continue
        # for dpath2clean in [os.path.join(checkpoints_dir, expname, adir) for adir in
        #                   ('checkpoints', 'checkpoints_test', 'vis')]:
        members = os.listdir(os.path.join(checkpoints_dir, expname, tests_dir))
        print('cleanup_checkpoints: {}: found {} models'.format(expname, len(members)))
        print('cleanup_checkpoints: DBG: keepers: {}'.format(keepers))
        for amember in members:
            rm = True
            for akeeper in keepers:
                if amember == akeeper:
                    rm = False
                    continue
            if rm:
                rmpath = os.path.join(checkpoints_dir, expname, tests_dir, amember)
                if os.path.isdir(rmpath):
                    shutil.rmtree(rmpath)
                    #pass
                # elif os.path.isdir(rmpath):
                #     #pass
                #     shutil.rmtree(rmpath)
                else:
                    raise ValueError(rmpath)
                print('rm -r '+rmpath)
    if len(empties) > 0:
        print('Empty models: {}'.format(empties))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cleanup_checkpoints(sys.argv[1])
    else:
        cleanup_checkpoints()
