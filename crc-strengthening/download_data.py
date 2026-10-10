"""Fetch the datasets the experiments need.

- CRC-VAL-HE-7K : the patient-disjoint external test set (Zenodo, ~700 MB).
- LC25000 (colon): an INDEPENDENT-SOURCE colon histology set for cross-dataset
  tumour-vs-benign validation. It is distributed on Kaggle; because Kaggle needs a
  token, this script tries the Kaggle CLI and otherwise prints manual instructions.

Usage:
    python download_data.py --out data            # gets CRC-VAL-HE-7K
    python download_data.py --out data --lc25000  # also tries LC25000 via kaggle
"""
import argparse, sys, zipfile, subprocess
from pathlib import Path
from urllib.request import urlopen

ZEN = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1"


def dl(url, dest):
    with urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0)); done = 0
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b); done += len(b)
                if total:
                    sys.stdout.write(f"\r  {dest.name}: {100*done/total:5.1f}%"); sys.stdout.flush()
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--lc25000", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = out / "CRC-VAL-HE-7K.zip"
    if not (out / "CRC-VAL-HE-7K").exists():
        if not z.exists():
            print("downloading CRC-VAL-HE-7K ...")
            dl(ZEN, z)
        print("extracting ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    print("CRC-VAL-HE-7K ready at", out / "CRC-VAL-HE-7K")

    if a.lc25000:
        target = out / "lc25000_colon"
        if target.exists():
            print("LC25000 colon already present at", target)
        else:
            print("\nAttempting LC25000 download via Kaggle CLI ...")
            try:
                subprocess.run(["kaggle", "datasets", "download", "-d",
                                "andrewmvd/lung-and-colon-cancer-histopathological-images",
                                "-p", str(out), "--unzip"], check=True)
                print("Kaggle download done. Locate the 'colon_image_sets' folder; it "
                      "contains colon_aca and colon_n. Point exp5 at that folder.")
            except Exception as e:
                print("Kaggle CLI not available or failed:", e)
                print(_LC_MANUAL)


_LC_MANUAL = """
----------------------------------------------------------------------
MANUAL LC25000 (colon) download
----------------------------------------------------------------------
1. On Kaggle: 'andrewmvd/lung-and-colon-cancer-histopathological-images'
   (or search 'LC25000'). Download and unzip.
2. You only need the COLON part. Arrange two folders:
        lc25000_colon/colon_aca/*.jpeg   (adenocarcinoma = tumour)
        lc25000_colon/colon_n/*.jpeg     (benign        = normal)
3. Pass that parent folder to exp5:
        python exp5_cross_dataset.py --root data/lc25000_colon --ckpt best_model.pt
On Colab you can also run:  !pip -q install kaggle  then upload kaggle.json.
Alternative independent set: Chaoyang (bupt-ai-cz/HSA-NRL) has normal/adenocarcinoma
from a different hospital; arrange its 'normal' and 'adenocarcinoma' folders the same way.
----------------------------------------------------------------------"""


if __name__ == "__main__":
    main()
