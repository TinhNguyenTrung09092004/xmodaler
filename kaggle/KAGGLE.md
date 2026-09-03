# Chạy COS-Net trên Kaggle (GPU T4 x2)

## 0. Thiếu gì

Dataset hiện tại của bạn có:

```
mscoco_caption_anno_clipfilter_fast_{train,val,test}.pkl
CLIP_RN101_49.tar.00{1,2,3,4}
```

Còn **thiếu 3 file bắt buộc**, không thể sinh ra từ những gì đang có:

| File | Config key | Dùng để làm gì |
|---|---|---|
| `vocabulary.txt` | `INFERENCE.VOCAB` | map token id trong pkl ngược lại thành chữ. Không có nó thì `load_vocab()` chết ngay lúc dựng `BeamSearcher`. |
| `captions_val5k.json` | `INFERENCE.VAL_ANNFILE` | ground truth COCO cho split val 5k (Karpathy) |
| `captions_test5k.json` | `INFERENCE.TEST_ANNFILE` | ground truth COCO cho split test 5k |

Cả 3 nằm trong thư mục `cosnet` trên Google Drive / Baidu ở
[configs/image_caption/cosnet/README.md](../configs/image_caption/cosnet/README.md)
(mã Baidu: `cosn`). Tải về, thêm vào cùng Kaggle dataset, xong.

Hai file cho giai đoạn RL — `mscoco_train_gts.pkl` và `mscoco_train_cider.pkl` —
thì **không cần tải**, `prepare_data.py --build-cider` sinh được từ file
`..._fast_train.pkl` qua `tools/cider_cache.py`.

## 1. Notebook cells

Bật GPU **T4 x2** + Internet trong Settings.

### Cell 1 — code + deps

```python
!git clone -b cosnet_kaggle <URL_REPO_CUA_BAN> /kaggle/working/xmodaler
%cd /kaggle/working/xmodaler
!bash kaggle/setup.sh
```

### Cell 2 — dựng data

```python
%cd /kaggle/working/xmodaler
!python kaggle/prepare_data.py --input-dir /kaggle/input/x-modaler-cosnet
```

Bỏ `--input-dir` thì script tự dò trong `/kaggle/input/**`. Script sẽ:

- symlink 3 file pkl vào `/kaggle/temp/open_source_dataset/mscoco_dataset/cosnet/`
  (code đọc annotation ở subfolder `cosnet/`, xem
  [mscoco_cosnet.py:53-57](../xmodaler/datasets/images/mscoco_cosnet.py#L53-L57))
- symlink `vocabulary.txt` + 2 file json vào data root
- `cat CLIP_RN101_49.tar.00* | tar -x` thẳng vào `/kaggle/temp/.../features/`
  (không ghép file trung gian, để khỏi tốn gấp đôi disk)
- kiểm tra `<image_id>.npz` khớp với `image_id` trong pkl và có đúng 2 key
  `features` / `g_feature`

Thêm `--build-cider` nếu định chạy RL sau đó.

### Cell 3 — train XE

```python
%cd /kaggle/working/xmodaler
!bash kaggle/run_train.sh
```

Session sau (checkpoint đã có sẵn trong `/kaggle/working/cosnet_output`):

```python
!bash kaggle/run_train.sh --resume
```

### Cell 4 (chạy song song) — dọn checkpoint

```python
!python kaggle/prune_ckpt.py --keep 2
```

## 2. Vì sao là 2 GPU x batch 16

`DATALOADER.TRAIN_BATCH_SIZE` là batch **mỗi process** (mỗi GPU) —
`build_xmodaler_train_loader` bọc `DistributedSampler` rồi mới đưa batch_size vào
`DataLoader`. Config gốc: 4 GPU x 8 = 32 ảnh/step, ~3540 iter/epoch, `NoamLR`
warmup 20000 iter (~5.6 epoch).

2 GPU x 16 = 32 ảnh/step → **iter/epoch và lịch warmup giữ nguyên y hệt bản gốc**,
không cần chỉnh `LR_SCHEDULER.WARMUP`.

Nếu OOM trên T4 16GB (mỗi ảnh nở thành `SEQ_PER_SAMPLE=5` câu → 80 chuỗi/GPU):

```bash
bash kaggle/run_train.sh DATALOADER.TRAIN_BATCH_SIZE 8 LR_SCHEDULER.WARMUP 40000
```

(batch giảm nửa → iter/epoch gấp đôi → warmup phải gấp đôi để giữ nguyên số epoch
warmup. Global batch lúc này là 16, thấp hơn paper.)

## 3. Vấn đề thời gian — đọc trước khi chạy

- 35 epoch x ~3540 iter. Trên T4 ước chừng **20–30 giờ**, trong khi 1 session
  Kaggle tối đa 12 giờ và quota GPU là 30 giờ/tuần. Tức là ~3 session, gần hết
  quota một tuần, **chưa tính** giai đoạn RL 60 epoch (còn nặng hơn nhiều vì mỗi
  step phải sample thêm).
- `cosnet_kaggle.yaml` đặt `TEST_EVAL_START: 29` để bỏ qua beam search trên test
  5k mỗi epoch; val vẫn chạy đủ. Muốn nhanh hơn nữa thì `SOLVER.EVAL_PERIOD 2`.
- `/kaggle/temp` **không** được giữ giữa các session → phải giải nén lại
  CLIP_RN101_49 mỗi lần (vài phút–vài chục phút). Nếu thấy phiền: giải nén một
  lần rồi upload thư mục `.npz` thành một Kaggle dataset thứ hai, mount trực tiếp
  và trỏ `DATALOADER.FEATS_FOLDER` vào đó — bỏ hẳn bước giải nén.

### Nối session

1. Session đang chạy: **Save Version → Save & Run All** để `/kaggle/working/cosnet_output`
   thành output được lưu.
2. Session mới: Add data → chọn output của notebook trước.
3. Copy về rồi resume:

```python
!mkdir -p /kaggle/working/cosnet_output
!cp /kaggle/input/<ten-output-truoc>/cosnet_output/last_checkpoint /kaggle/working/cosnet_output/
!cp /kaggle/input/<ten-output-truoc>/cosnet_output/model_Epoch_*.pth /kaggle/working/cosnet_output/
!bash kaggle/run_train.sh --resume
```

`last_checkpoint` chỉ chứa basename nên copy sang thư mục khác vẫn resume đúng.

## 4. Giai đoạn RL

```python
!python kaggle/prepare_data.py --input-dir ... --skip-features --build-cider
!cp /kaggle/working/cosnet_output/model_Epoch_00035_Iter_*.pth /kaggle/working/cosnet_output/cosnet_xe.pth
!bash kaggle/run_train_rl.sh
```

Đổi `MODEL.WEIGHTS` trong `cosnet_rl_kaggle.yaml` nếu đặt tên file khác.

## 5. Lỗi hay gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| `FileNotFoundError: .../vocabulary.txt` | thiếu file ở mục 0 |
| `java: command not found` khi eval | `apt-get update && apt-get install -y default-jre`; `pycocoevalcap` gọi java cho PTBTokenizer/METEOR/SPICE |
| `No module named 'pytorch_transformers'` | `xmodaler/tokenization` import nó lúc load module dù COS-Net không dùng BERT tokenizer — `setup.sh` đã cài |
| `No space left on device` giữa chừng | disk Kaggle ~57GB dùng chung; `prepare_data.py` check trước khi giải nén, nhưng checkpoint tích lũy cũng ăn dần → chạy `prune_ckpt.py` |
| DataLoader treo / worker chết | Kaggle chỉ có 4 vCPU; giữ `NUM_WORKERS: 2` (config gốc để 6, x2 process là quá tải) |
