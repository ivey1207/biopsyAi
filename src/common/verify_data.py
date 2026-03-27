from pathlib import Path


def count_files(path: Path, ext: str):
    return len(list(path.glob(f"*{ext}")))


def main():
    base = Path("data")
    checks = [
        ("classification/test", ".png", 1276),
        ("classification/train/0", ".png", 571),
        ("classification/train/1", ".png", 974),
        ("classification/train/2", ".png", 1043),
        ("classification/train/3", ".png", 750),
        ("classification/train/4", ".png", 814),
        ("classification/train/5", ".png", 441),
        ("classification/train/6", ".png", 545),
        ("classification/train/7", ".png", 2136),
        ("classification/train/8", ".png", 331),
        ("classification/train/9", ".png", 1111),
        ("classification/train/10", ".png", 899),
        ("classification/train/11", ".png", 1796),
        ("Segmentation/training/images", ".jpg", 1800),
        ("Segmentation/training/masks", ".png", 1800),
        ("Segmentation/validation/images", ".jpg", 400),
        ("Segmentation/validation/masks", ".png", 400),
        ("Segmentation/testing/images", ".jpg", 200),
    ]
    ok = True
    for rel, ext, expected in checks:
        p = base / rel
        got = count_files(p, ext)
        status = "OK" if got == expected else "MISMATCH"
        print(f"{status:9} {rel:35} got={got:5} expected={expected:5}")
        ok = ok and got == expected
    print("\nRESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
