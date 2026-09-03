# Chạy COS-Net trên Kaggle (GPU T4 x2)

Settings: Accelerator `GPU T4 x2`, Internet `On`.

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
!bash kaggle/run_train.sh INFERENCE.VAL_EVAL_START 24
```

Session sau, checkpoint đã có sẵn trong `/kaggle/working/cosnet_output`:

```python
!bash kaggle/run_train.sh --resume INFERENCE.VAL_EVAL_START 24
```

Override đó là để tăng tốc, xem mục 3. Bỏ đi vẫn chạy đúng, chỉ chậm hơn.

## 2. Vì sao là 2 GPU x batch 16

`DATALOADER.TRAIN_BATCH_SIZE` là batch **mỗi process** (mỗi GPU) —
`build_xmodaler_train_loader` bọc `DistributedSampler` rồi mới đưa batch_size vào
`DataLoader`. Config gốc: 4 GPU x 8 = 32 ảnh/step, ~3540 iter/epoch, `NoamLR`
warmup 20000 iter (~5.6 epoch).

2 GPU x 16 = 32 ảnh/step → iter/epoch và lịch warmup giữ nguyên y hệt bản gốc,
không cần chỉnh `LR_SCHEDULER.WARMUP`.

Thực đo: `max_mem: 3900M` / 15GB, nên OOM không phải vấn đề ở mức batch này.

## 3. Thời gian — đọc trước khi chạy

Thực đo epoch 1: `time: 1.0090  data_time: 0.5320`, ETA **36 giờ** cho 35 epoch.
Con số ETA đó **chỉ tính iter train**, chưa cộng eval. Session Kaggle tối đa 12h,
quota 30h/tuần → cần 3-4 session, và đó là chưa tính RL 60 epoch.

Hai cách rút ngắn, không đổi chất lượng mô hình:

- **`INFERENCE.VAL_EVAL_START 24`** (override lúc chạy) — mỗi epoch `EvalHook` beam-search
  5000 ảnh val (~8-10 phút), × 35 epoch ≈ 5 tiếng. Kết quả đó không feed ngược vào đâu cả:
  [defaults.py:356-390](../xmodaler/engine/defaults.py#L356-L390) chỉ log ra, không chọn
  best checkpoint, không early stopping. Bỏ eval sớm → trọng số cuối giống hệt.
- **`DATALOADER.NUM_WORKERS: 4`** (đã đặt sẵn trong `cosnet_kaggle.yaml`) — một nửa mỗi
  step là ngồi chờ dữ liệu. `data_time 0.53` cho batch 16 = ~33ms/file `.npz`, tức độ trễ
  mở file nhỏ, không phải băng thông (12.8MB/s) cũng không phải CPU. Nên dù Kaggle chỉ có
  4 vCPU, tăng worker vẫn ăn vì chúng nằm chờ I/O. Nếu `data_time` về gần 0 thì ETA còn ~17h.

Đo lại sau ~100 iter để biết lãi bao nhiêu.

`TEST_EVAL_START: 29` đã có sẵn trong `cosnet_kaggle.yaml` để bỏ beam search trên
test 5k mỗi epoch.

`/kaggle/temp` **không** được giữ giữa các session → phải giải nén lại
CLIP_RN101_49 mỗi lần. Muốn bỏ hẳn: upload thư mục `.npz` đã giải thành một Kaggle
dataset thứ hai rồi trỏ `DATALOADER.FEATS_FOLDER` vào đó.

### Nối session

1. Session đang chạy: **Save Version → Save & Run All** để `/kaggle/working/cosnet_output`
   thành output được lưu.
2. Session mới: Add data → chọn output của notebook trước.
3. Copy về rồi resume:

```python
!mkdir -p /kaggle/working/cosnet_output
!cp /kaggle/input/<ten-output-truoc>/cosnet_output/last_checkpoint /kaggle/working/cosnet_output/
!cp /kaggle/input/<ten-output-truoc>/cosnet_output/model_Epoch_*.pth /kaggle/working/cosnet_output/
!bash kaggle/run_train.sh --resume INFERENCE.VAL_EVAL_START 24
```

`last_checkpoint` chỉ chứa basename nên copy sang thư mục khác vẫn resume đúng.

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
