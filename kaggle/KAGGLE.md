# Running COS-Net on Kaggle (GPU T4 x2)

## 1. Notebook cells

### Cell 1

```python
!git clone -b cosnet_kaggle https://github.com/TinhNguyenTrung09092004/xmodaler.git /kaggle/working/xmodaler
%cd /kaggle/working/xmodaler
!bash kaggle/setup.sh
```

### Cell 2

```python
%cd /kaggle/working/xmodaler
!python kaggle/prepare_data.py
```

### Cell 3

```python
%cd /kaggle/working/xmodaler
!bash kaggle/run_train.sh
```

## 2. Resuming a session

```python
%cd /kaggle/working/xmodaler
!mkdir -p /kaggle/working/cosnet_output
!echo "/kaggle/input/<slug>/model.pth" > /kaggle/working/cosnet_output/last_checkpoint
!bash kaggle/run_train.sh --resume
```

Check the `Loading checkpoint from ...` line in the log to make sure it is the right file; the starting iteration must equal the number in the filename + 1.
