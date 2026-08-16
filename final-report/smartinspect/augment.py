from pathlib import Path
from PIL import Image, ImageEnhance

folders = [
    "good",
    "thread_top",
    "thread_side",
    "scratch_head",
    "scratch_neck",
    "manipulated_front"
]

base = Path("dataset/train")

total = 0

for folder in folders:

    target = base / folder

    print(f"\n処理中: {folder}")

    for img_path in target.iterdir():

        if img_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        try:

            img = Image.open(img_path)

            img.transpose(
                Image.FLIP_LEFT_RIGHT
            ).save(
                target / f"{img_path.stem}_flip.png"
            )

            img.rotate(
                10,
                expand=True
            ).save(
                target / f"{img_path.stem}_r10.png"
            )

            img.rotate(
                -10,
                expand=True
            ).save(
                target / f"{img_path.stem}_r_10.png"
            )

            ImageEnhance.Brightness(
                img
            ).enhance(
                1.2
            ).save(
                target /
                f"{img_path.stem}_bright.png"
            )

            total += 4

        except Exception as e:

            print(
                f"エラー: {img_path}"
            )
            print(e)

print(f"\n追加生成枚数: {total}")