# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_face_eval
from model.config import cfg

class wider_face(imdb):
  def __init__(self, image_set, use_diff=False):
    name = 'wider_face_'+image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
#    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    self.img_root = self.get_image_root_dir()
    self._data_path = os.path.join(self._devkit_path, 'wider_face_split')
#    == self._devkit_path?
#    +'wider_face_split'?
    self._classes = ('__background__',  # always index 0
                     'face')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'
    
    if self._image_set == 'train':
      filename = os.path.join(self._data_path, self._image_set+'_bbox_dict.npy')
      self.bbox_dict = np.load(filename)
    elif  self._image_set == 'val':
      filename = os.path.join(self._data_path, self._image_set+'_bbox_dict.npy')
      self.bbox_dict = np.load(filename)
    
    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'wider_face path does not exist: {}'.format(self._devkit_path)
#    assert os.path.exists(self._data_path), \
#      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def find_image_root(self, img_name):
      
    img_ref = img_name.split('_')[0]
    return self.img_root[img_ref]

  def get_image_root_dir(self):
    root = os.path.join(self._devkit_path, 'WIDER_'+self._image_set, 'images')
    folders = [f for f in os.listdir(root)]
  
    img_folder_dir = {}
    for i in folders:
      tmp = i.split('--')
      img_folder_dir[tmp[0]]=os.path.join(root,i)

    return img_folder_dir
  
  def image_path_from_index(self, img_name):
    """
    Construct an image path from the image's "index" identifier.
    """
    
    image_path = os.path.join(self.find_image_root(img_name), img_name + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the image indeces listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /wider_face_split/val_file_list.npy
    image_set_file = os.path.join(self._data_path, self._image_set + '_file_list' + '.npy')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    image_index_file = np.load(image_set_file)
    image_index = []
    for i in image_index_file:
        image_index.extend(i)
    image_index = np.array(image_index)
    return image_index

  def _get_default_path(self):
    """
    Return the default path where wider_face is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'wider_face')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb', encoding="utf-8") as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)



  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from txt file in the wider_face
    format.
    """
#    assert self.bbox_dict is not None
    objs = self.bbox_dict.item().get(index)
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      
      # Make pixel indexes 0-based
      x1 = obj[0]
      y1 = obj[1]
      x2 = obj[0] + obj[2]
      y2 = obj[1] + obj[3]
#      print('boxes---------------')
#      print(x1,y1,x2,y2)
#      print('boxes---------------')
      cls = 1
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    
    path = os.path.join(
      self._devkit_path,
      'results',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt', encoding="utf-8") as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
#    annopath = os.path.join(
#      self._devkit_path,
#      'VOC' + self._year,
#      'Annotations',
#      '{:s}.xml')
#    imagesetfile = os.path.join(
#      self._devkit_path,
#      'VOC' + self._year,
#      'ImageSets',
#      'Main',
#      self._image_set + '.txt')
#    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    gt_path = os.path.join(self._devkit_path,'wider_face_split')
    aps = []
    # The PASCAL VOC metric changed in 2010
#    use_07_metric = True if int(self._year) < 2010 else False
    use_07_metric = True
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_face_eval(filename, gt_path, cls, ovthresh=0.5,use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
#    print(all_boxes[0])
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.wider_face import wider_face

  d = wider_face()#'trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
