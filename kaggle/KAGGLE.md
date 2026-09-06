# Chạy COS-Net trên Kaggle (GPU T4 x2)

## 1. Notebook cells

### Cell 1 — code + deps

```python
!git clone -b cosnet_kaggle https://github.com/TinhNguyenTrung09092004/xmodaler.git /kaggle/working/xmodaler
%cd /kaggle/working/xmodaler
!bash kaggle/setup.sh
```

### Cell 2 — dựng data

```python
%cd /kaggle/working/xmodaler
!python kaggle/prepare_data.py
```

Thêm `--build-cider` nếu định chạy RL sau đó.

### Cell 3 — train XE

```python
%cd /kaggle/working/xmodaler
!bash kaggle/run_train.sh INFERENCE.VAL_EVAL_START 24 SOLVER.EPOCH 1
```

## 2. Vì sao là 2 GPU x batch 16

`DATALOADER.TRAIN_BATCH_SIZE` là batch **mỗi process** (mỗi GPU). Config gốc: 4 GPU x 8 = 32 ảnh/step, ~3540 iter/epoch, `NoamLR`
warmup 20000 iter (~5.6 epoch).

2 GPU x 16 = 32 ảnh/step → iter/epoch và lịch warmup giữ nguyên y hệt bản gốc,
không cần chỉnh `LR_SCHEDULER.WARMUP`.

## 3. Nối session

```python
%cd /kaggle/working/xmodaler
!mkdir -p /kaggle/working/cosnet_output
!echo "/kaggle/input/<slug>/model_Epoch_00005_Iter_0017699.pth" > /kaggle/working/cosnet_output/last_checkpoint
!bash kaggle/run_train.sh --resume INFERENCE.VAL_EVAL_START 24
```

Kiểm log dòng `Loading checkpoint from ...` để chắc đúng file; iter bắt đầu phải
bằng số trong tên file + 1.

## 4. Giai đoạn RL

```python
!python kaggle/prepare_data.py --skip-features --build-cider
!cp /kaggle/working/cosnet_output/model_Epoch_00035_Iter_*.pth /kaggle/working/cosnet_output/cosnet_xe.pth
!bash kaggle/run_train_rl.sh
```

Đổi `MODEL.WEIGHTS` trong `cosnet_rl_kaggle.yaml` nếu đặt tên file khác.

`cosnet_rl_kaggle.yaml` đặt `SOLVER.FIND_UNUSED_PARAMETERS: True` vì nhánh RL chạy
thêm một lượt `no_grad` + sampling decode mà chưa quan sát được. Nếu DDP không kêu
thì bỏ đi để nhanh hơn vài %.
