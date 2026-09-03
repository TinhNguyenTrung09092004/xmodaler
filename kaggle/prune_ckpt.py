import argparse
import os
import re
import glob

EPOCH_RE = re.compile(r"_Epoch_(\d+)_Iter_(\d+)\.pth$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/kaggle/working/cosnet_output")
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpts = []
    for path in glob.glob(os.path.join(args.output_dir, "*.pth")):
        m = EPOCH_RE.search(os.path.basename(path))
        if m:
            ckpts.append((int(m.group(2)), path))
    ckpts.sort()

    protected = set()
    last = os.path.join(args.output_dir, "last_checkpoint")
    if os.path.exists(last):
        with open(last) as f:
            protected.add(os.path.join(args.output_dir, f.read().strip()))

    doomed = [p for _, p in ckpts[:-args.keep]] if args.keep > 0 else [p for _, p in ckpts]
    freed = 0
    for path in doomed:
        if path in protected:
            continue
        size = os.path.getsize(path)
        print("%s %s (%.2f GB)" % ("would remove" if args.dry_run else "removing",
                                   os.path.basename(path), size / 1024 ** 3))
        if not args.dry_run:
            os.remove(path)
        freed += size

    kept = [p for _, p in ckpts if p not in doomed or p in protected]
    print("kept %d checkpoint(s), freed %.2f GB" % (len(kept), freed / 1024 ** 3))
    for p in kept:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
