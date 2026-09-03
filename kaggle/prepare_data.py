"""
Build the directory layout that configs/image_caption/cosnet expects out of the
files in a read-only Kaggle input dataset.

    DATA_ROOT/
      cosnet/mscoco_caption_anno_clipfilter_fast_{train,val,test}.pkl
      vocabulary.txt
      captions_val5k.json
      captions_test5k.json
      mscoco_train_gts.pkl        (--build-cider)
      mscoco_train_cider.pkl      (--build-cider)
      features/CLIP_RN101_49/<image_id>.npz

Run from the repo root:
    python kaggle/prepare_data.py
    python kaggle/prepare_data.py --build-cider
"""
import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys

DEFAULT_DATA_ROOT = "/kaggle/temp/open_source_dataset/mscoco_dataset"
FEATS_DIRNAME = "CLIP_RN101_49"

ANNO_FILES = [
    "mscoco_caption_anno_clipfilter_fast_train.pkl",
    "mscoco_caption_anno_clipfilter_fast_val.pkl",
    "mscoco_caption_anno_clipfilter_fast_test.pkl",
]
ROOT_FILES = [
    "vocabulary.txt",
    "captions_val5k.json",
    "captions_test5k.json",
]


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return "%.1f%s" % (n, unit)
        n /= 1024.0
    return "%.1fPB" % n


def sh(cmd):
    return subprocess.call(["bash", "-c", "set -o pipefail; " + cmd])


def find_input_dir(explicit):
    if explicit:
        if not os.path.isdir(explicit):
            sys.exit("[FATAL] --input-dir %s does not exist" % explicit)
        return explicit
    hits = glob.glob("/kaggle/input/**/" + ANNO_FILES[0], recursive=True)
    if not hits:
        sys.exit("[FATAL] could not auto-detect the input dataset. Pass --input-dir "
                 "(the folder holding %s)" % ANNO_FILES[0])
    return os.path.dirname(hits[0])


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            print("  ok (already linked) %s" % dst)
            return
        os.remove(dst)
    os.symlink(src, dst)
    print("  linked %s -> %s" % (dst, src))


def locate_npz_dir(feats_root):
    """Return the directory actually holding the .npz files, whatever the tar layout."""
    for path in glob.iglob(os.path.join(feats_root, "**", "*.npz"), recursive=True):
        return os.path.dirname(path)
    return None


def extract_features(input_dir, data_root, force):
    feats_root = os.path.join(data_root, "features")
    feats_dir = os.path.join(feats_root, FEATS_DIRNAME)
    os.makedirs(feats_root, exist_ok=True)

    parts = sorted(glob.glob(os.path.join(input_dir, FEATS_DIRNAME + ".tar.*")))
    if not parts:
        sys.exit("[FATAL] no %s.tar.* parts found in %s" % (FEATS_DIRNAME, input_dir))

    found = locate_npz_dir(feats_root)
    if found and not force:
        print("[feats] .npz files already present in %s, skipping extraction "
              "(--force-extract to redo)" % found)
        return found

    total = sum(os.path.getsize(p) for p in parts)
    free = shutil.disk_usage(data_root).free
    print("[feats] %d parts, %s total" % (len(parts), human(total)))
    print("[feats] free space on %s: %s" % (data_root, human(free)))
    if free < total * 1.05:
        sys.exit("[FATAL] not enough disk: need ~%s free, have %s" % (human(total), human(free)))

    cat_parts = " ".join("'%s'" % p for p in parts)
    print("[feats] archive layout:")
    sh("cat %s | tar -tf - 2>/dev/null | head -5" % cat_parts)

    print("[feats] extracting into %s ..." % feats_root)
    if sh("cat %s | tar -x -C '%s'" % (cat_parts, feats_root)) != 0:
        sys.exit("[FATAL] extraction failed")

    found = locate_npz_dir(feats_root)
    if found is None:
        print("[FATAL] no *.npz anywhere under %s. What was extracted:" % feats_root)
        sh("find '%s' -maxdepth 3 | head -30" % feats_root)
        sys.exit(1)

    if os.path.realpath(found) != os.path.realpath(feats_dir):
        # The config points at features/CLIP_RN101_49; bridge whatever tar produced.
        print("[feats] .npz files landed in %s" % found)
        if os.path.islink(feats_dir):
            os.remove(feats_dir)
        if not os.path.exists(feats_dir):
            os.symlink(found, feats_dir)
            print("[feats] linked %s -> %s" % (feats_dir, found))
    return feats_dir


def verify_features(feats_dir, anno_path):
    import numpy as np

    n = len(glob.glob(os.path.join(feats_dir, "*.npz")))
    print("[verify] %d .npz files visible at %s" % (n, feats_dir))

    datalist = pickle.load(open(anno_path, "rb"), encoding="bytes")
    missing = [d["image_id"] for d in datalist[:200]
               if not os.path.exists(os.path.join(feats_dir, d["image_id"] + ".npz"))]
    if missing:
        print("[FATAL] no feature file for image_ids like %s. Sample of what is there:"
              % missing[:5])
        sh("ls '%s' | head -5" % feats_dir)
        sys.exit(1)

    sample = os.path.join(feats_dir, datalist[0]["image_id"] + ".npz")
    with np.load(sample) as c:
        keys = list(c.files)
        if "features" not in keys or "g_feature" not in keys:
            sys.exit("[FATAL] %s has keys %s, expected 'features' and 'g_feature'"
                     % (sample, keys))
        print("[verify] %s: features%s g_feature%s"
              % (os.path.basename(sample), c["features"].shape, c["g_feature"].shape))
    print("[verify] OK")


def build_cider(data_root, repo_root):
    anno = os.path.join(data_root, "cosnet", ANNO_FILES[0])
    ids_file = os.path.join(data_root, "coco_train_image_id.txt")
    gts = os.path.join(data_root, "mscoco_train_gts.pkl")
    cider = os.path.join(data_root, "mscoco_train_cider.pkl")

    datalist = pickle.load(open(anno, "rb"), encoding="bytes")
    with open(ids_file, "w") as f:
        for d in datalist:
            f.write("%s\n" % d["image_id"])
    print("[cider] wrote %d train image ids to %s" % (len(datalist), ids_file))

    rc = subprocess.call([
        sys.executable, os.path.join(repo_root, "tools", "cider_cache.py"),
        "--infile", anno, "--outfile", cider, "--gts", gts, "--image_ids", ids_file,
    ])
    if rc != 0:
        sys.exit("[FATAL] tools/cider_cache.py failed (exit %d)" % rc)
    print("[cider] wrote %s and %s" % (gts, cider))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="must match ANNO_FOLDER in kaggle/cosnet_kaggle.yaml")
    ap.add_argument("--build-cider", action="store_true",
                    help="generate mscoco_train_gts.pkl / mscoco_train_cider.pkl for the RL stage")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = find_input_dir(args.input_dir)
    data_root = args.data_root
    print("[paths] input_dir = %s" % input_dir)
    print("[paths] data_root = %s" % data_root)

    os.makedirs(os.path.join(data_root, "cosnet"), exist_ok=True)
    os.makedirs(os.path.join(repo_root, "data", "temp"), exist_ok=True)

    print("[anno] linking annotation pkl files")
    for name in ANNO_FILES:
        src = os.path.join(input_dir, name)
        if not os.path.exists(src):
            sys.exit("[FATAL] missing %s in %s" % (name, input_dir))
        link(src, os.path.join(data_root, "cosnet", name))

    print("[root] linking vocabulary / evaluation ground truth")
    missing = []
    for name in ROOT_FILES:
        src = os.path.join(input_dir, name)
        if os.path.exists(src):
            link(src, os.path.join(data_root, name))
        else:
            missing.append(name)
    if missing:
        sys.exit("[FATAL] missing from %s: %s" % (input_dir, ", ".join(missing)))

    if not args.skip_features:
        feats_dir = extract_features(input_dir, data_root, args.force_extract)
        verify_features(feats_dir, os.path.join(data_root, "cosnet", ANNO_FILES[0]))

    if args.build_cider:
        build_cider(data_root, repo_root)

    print("\n[done] data root ready at %s" % data_root)


if __name__ == "__main__":
    main()
