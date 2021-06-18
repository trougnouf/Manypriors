# -*- coding: utf-8 -*-
import mwclient
import json
import os
import random
import logging
import argparse
import sys
sys.path.append('..')
from common.libs import utilities
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
#https://commons.wikimedia.org/wiki/Special:ApiSandbox#action=query&format=json&prop=&list=categorymembers&cmtitle=Category%3AFeatured%20pictures&cmtype=subcat|file&cmlimit=max

LIMIT = {'FP': None, 'QI': 50000}
YEAR = {'FP': None, 'QI': 2018}
ROOTDIR = os.path.join('..', '..', 'datasets')
DLDIR = {'FP': 'FeaturedPictures',
         'QI': 'QualityImages%u'%YEAR['QI']}
CAT = {'FP': 'Category:Featured_pictures_on_Wikimedia_Commons', 'QI': 'Category:Quality_images'}
JSONDIR = 'cats_content'
EXTS_TO_SKIP = 'webm', 'svg', 'ogv'
FORCE_REFRESH_CATS = True

SITE = mwclient.Site('commons.wikimedia.org', clients_useragent='TrougnoufDownloader/0.1 (b.brummer@intopix.com)')

def get_cat_imgs(rootcat=CAT['FP'], site=SITE, known_cats=set(), subcats=True) -> list:
    json_fpath = os.path.join(JSONDIR, rootcat.replace('/', '-')+'.json')
    if os.path.exists(json_fpath) and not FORCE_REFRESH_CATS:
        with open(json_fpath, 'r') as fp:
            return json.load(fp)
    print('Searching %s' % rootcat)
    res = set()
    known_cats.add(rootcat)
    if subcats:
        subcatsres = site.api('query', list='categorymembers', cmtitle=rootcat, cmtype='subcat', cmlimit='max')['query']['categorymembers']
        for acat in subcatsres:
            if acat['title'] in known_cats:
                continue
            res.update(get_cat_imgs(rootcat=acat['title']))
    cmcontinue = queryres = None
    while cmcontinue is not None or queryres is None:
        queryres = site.api('query', list='categorymembers', cmtitle=rootcat, cmtype='file', cmlimit='max', cmcontinue=cmcontinue)#['query']['categorymembers']
        for afile in queryres['query']['categorymembers']:
            res.add(afile['title'])
        cmcontinue = None if 'continue' not in queryres else queryres['continue']['cmcontinue']
    with open(json_fpath, 'w') as fp:
        json.dump(list(res), fp)
    return res


def dl_cat(rootcat=CAT['FP'], site=SITE, subcats=True, limit=LIMIT['FP'], dlrootdir=ROOTDIR, dldir=DLDIR['FP'], year=YEAR['FP'], delete_uncat=False):
    dl_dpath = os.path.join(dlrootdir, dldir)
    os.makedirs(dl_dpath, exist_ok=True)
    todl_pictures = list(get_cat_imgs(rootcat=rootcat, subcats=subcats))
    random.shuffle(todl_pictures)
    if delete_uncat:
        for fn in os.listdir(dl_dpath):
            if 'File:'+fn not in todl_pictures:
                todel_fpath = os.path.join(dl_dpath, fn)
                print(f'rm {todel_fpath}')
                os.remove(todel_fpath)
    for i, apic in enumerate(tqdm(todl_pictures), start=1):
        if apic.split('.')[-1] in EXTS_TO_SKIP:
            continue
        target_fpath = os.path.join(dl_dpath, apic.split(':')[-1])
        im = site.images[apic.split(':')[-1]]
        checksum = im.imageinfo['sha1']
        if os.path.exists(target_fpath):
            localhash = utilities.checksum(target_fpath, htype='sha1')
            if checksum == localhash:
                continue
            else:
                print(f'warning: checksums do not match with local file. Re-downloading {target_fpath}.')
        if year is not None:
            try:
                imyear = im._info['imageinfo'][0]['timestamp'][0:4]
            except KeyError:
                breakpoint()
            if imyear != str(year):
                continue
        with open(target_fpath, 'wb') as fp:
            try:
                print("Downloading %s" % apic)
                im.download(fp)
            except Exception as e:
                print('Unable to download %s: %s' % (apic, e))
        localhash = utilities.checksum(target_fpath, htype='sha1')
        if localhash != checksum:
            print(f'warning: checksums do not match; {target_fpath} is likely corrupted. You can download it manually or start this script over.')
            print(f'rm {target_fpath}')
            os.remove(target_fpath)
        if limit is not None and i == limit:
            break
        # cleanup decat images


def parser_add_arguments(parser):
    parser.add_argument('--cat', default='FP', help='Category short name (presets). default: FP, options: {}'.format(CAT.keys()))
    parser.add_argument('--category', help='Category long name (what follows Category: on Wikimedia Commons)')
    parser.add_argument('--dest_dirname', help='Destination directory name (default: Category:<CATEGORY>))')
    parser.add_argument('--limit', type=int, help='default: {}'.format(LIMIT))
    parser.add_argument('--year', type=int, help='default: {}'.format(YEAR))
    parser.add_argument('--rootdir', default=ROOTDIR, help='Root download directory (default: {})'.format(ROOTDIR))
    parser.add_argument('--delete_uncat', action='store_true', help='Delete images which are no longer part of the category')

def parser_autocomplete(args):
    '''
    args.cat presets
    '''
    if args.category is None:
        assert args.cat in CAT
        args.category = CAT[args.cat]
        if args.limit is None:
            args.limit = LIMIT[args.cat]
        if args.year is None:
            args.year = YEAR[args.cat]
        if args.dest_dirname is None:
            args.dest_dirname = DLDIR[args.cat]
    if args.dest_dirname is None:
        args.dest_dirname = args.category

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser_add_arguments(parser)
    args = parser.parse_args()
    parser_autocomplete(args)
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(JSONDIR, exist_ok=True)
    dl_cat(rootcat=args.category, limit=args.limit, dlrootdir=args.rootdir, dldir=args.dest_dirname, year=args.year, delete_uncat=args.delete_uncat)
    #dl_cat(rootcat=CAT['FP'], dldir=DLDIR['FP'], limit=LIMIT['FP'])
    #dl_cat(rootcat=CAT['QI'], dldir=DLDIR['QI'], limit=LIMIT['QI'], subcats=False, year=YEAR['QI'])
