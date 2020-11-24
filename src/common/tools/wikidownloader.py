# -*- coding: utf-8 -*-
import mwclient
import json
import os
import random
import logging
import argparse
logging.basicConfig(level=logging.INFO)
#https://commons.wikimedia.org/wiki/Special:ApiSandbox#action=query&format=json&prop=&list=categorymembers&cmtitle=Category%3AFeatured%20pictures&cmtype=subcat|file&cmlimit=max

LIMIT = {'FPC': None, 'QI': 50000}
YEAR = {'FPC': None, 'QI': 2018}
ROOTDIR = os.path.join('..', '..', 'datasets')
DLDIR = {'FPC': 'FeaturedPictures',
         'QI': 'QualityImages%u'%YEAR['QI']}
CAT = {'FPC': 'Category:Featured_pictures_on_Wikimedia_Commons', 'QI': 'Category:Quality_images'}
JSONDIR = 'cats_content'

SITE = mwclient.Site('commons.wikimedia.org', clients_useragent='TrougnoufDownloader/0.1 (b.brummer@intopix.com)')

def get_cat_imgs(rootcat=CAT['FPC'], site=SITE, known_cats=set(), subcats=True):
    json_fpath = os.path.join(JSONDIR, rootcat.replace('/', '-')+'.json')
    if os.path.exists(json_fpath):
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


def dl_cat(rootcat=CAT['FPC'], site=SITE, subcats=True, limit=LIMIT['FPC'], dlrootdir=ROOTDIR, dldir=DLDIR['FPC'], year=YEAR['FPC']):
    dl_dpath = os.path.join(dlrootdir, dldir)
    os.makedirs(dl_dpath, exist_ok=True)
    todl_pictures = list(get_cat_imgs(rootcat=rootcat, subcats=subcats))
    random.shuffle(todl_pictures)
    for i, apic in enumerate(todl_pictures, start=1):
        target_fpath = os.path.join(dl_dpath, apic.split(':')[-1])
        if os.path.exists(target_fpath):
            continue
        im = site.images[apic.split(':')[-1]]
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
        if limit is not None and i == limit:
            break

def parser_add_arguments(parser):
    parser.add_argument('--cat', default='FPC', help='Category short name. default: FPC, options: {}'.format(CAT.keys()))
    parser.add_argument('--category', help='Category long name (what follows Category: on Wikimedia Commons)')
    parser.add_argument('--limit', type=int, help='default: {}'.format(LIMIT))
    parser.add_argument('--year', type=int, help='default: {}'.format(YEAR))
    parser.add_argument('--rootdir', default=ROOTDIR, help='Root download directory (default: {})'.format(ROOTDIR))

def parser_autocomplete(args):
    if args.category is None:
        assert args.cat in CAT
        args.category = CAT[args.cat]
    if args.limit is None:
        assert args.cat in CAT
        args.limit = LIMIT[args.cat]
    if args.year is None:
        assert args.cat in CAT
        args.cat = YEAR[args.cat]

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
    dl_cat(rootcat=args.category, limit=args.limit, dlrootdir=args.rootdir, dldir=args.category, year=args.year)
    #dl_cat(rootcat=CAT['FPC'], dldir=DLDIR['FPC'], limit=LIMIT['FPC'])
    #dl_cat(rootcat=CAT['QI'], dldir=DLDIR['QI'], limit=LIMIT['QI'], subcats=False, year=YEAR['QI'])
