# Use Builtin Datasets

A dataset can be used by wrapping it into a torch Dataset. This document explains how to setup the builtin datasets so they can be used by X-modaler. The annotations for builtin datasets can be downloaded [here](https://drive.google.com/drive/folders/1vx9n7tAIt8su0y_3tsPJGvMPBMm8JLCZ).

X-modaler has builtin supports for the MSCOCO image captioning dataset. The corresponding dataset wrapper is provided in `xmodaler/datasets`:
```
xmodaler/datasets/
  images/
    mscoco.py
```
You can specify which dataset wrapper to use by `DATASETS.TRAIN`, `DATASETS.VAL` and `DATASETS.TEST` in the config file. 

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
mscoco_dataset/
  mscoco_caption_anno_train.pkl
  mscoco_caption_anno_val.pkl
  mscoco_caption_anno_test.pkl
  vocabulary.txt
  captions_val5k.json
  captions_test5k.json
  # image files that are mentioned in the corresponding json
  features/
    CLIP_RN101_49/
      *.npz
```

When the dataset wrapper and data files are ready, you need to specify the corresponding paths to these data files in the config file. For example, 
```
DATALOADER:
	FEATS_FOLDER: '../open_source_dataset/mscoco_dataset/features/CLIP_RN101_49'    # feature folder
	ANNO_FOLDER: '../open_source_dataset/mscoco_dataset' # annotation folders
INFERENCE:
	VOCAB: '../open_source_dataset/mscoco_dataset/vocabulary.txt'
	VAL_ANNFILE: '../open_source_dataset/mscoco_dataset/captions_val5k.json'
	TEST_ANNFILE:  '../open_source_dataset/mscoco_dataset/captions_test5k.json'
```
