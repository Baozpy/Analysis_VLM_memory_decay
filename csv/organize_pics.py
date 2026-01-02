import os
import csv
import shutil
from glob import glob

base_dir = "."
csv_dir = base_dir
pics_dir = os.path.join(base_dir, "mmdu_pics")

# 额外补充 sample100~109 的图片列表
extra_images = {
    "sample100": [
        "ILSVRC2012_val_00001915.JPEG",
        "ILSVRC2012_val_00002668.JPEG",
        "ILSVRC2012_val_00009092.JPEG",
        "ILSVRC2012_val_00005848.JPEG",
        "ILSVRC2012_val_00002929.JPEG",
        "ILSVRC2012_val_00014328.JPEG",
        "ILSVRC2012_val_00000439.JPEG",
        "ILSVRC2012_val_00003279.JPEG",
        "ILSVRC2012_val_00005535.JPEG",
        "aussie-bird-count-background-image.jpg.auspostimage.970_0.medium.jpg",
        "Indian_Peafowl.jpg",
        "Purple_Finch-Linda_Peresen-BS-FI-480x359.jpg",
        "chick-8116385_1280.jpg",
        "5690.jpg_wh300.jpg",
        "960x0.jpg"
    ],
    "sample101": [
        "3186969.jpg","1682911.jpg","1022479.jpg","1766448.jpg","3403012.jpg",
        "408666.jpg","682627.jpg","3610775.jpg","3572271.jpg","1814986.jpg","1804724.jpg"
    ],
    "sample102": [
        "ILSVRC2012_val_00000552.JPEG","ILSVRC2012_val_00002669.JPEG","ILSVRC2012_val_00004840.JPEG",
        "ILSVRC2012_val_00009267.JPEG","ILSVRC2012_val_00003402.JPEG","ILSVRC2012_val_00001705.JPEG",
        "ILSVRC2012_val_00004978.JPEG","ILSVRC2012_val_00002815.JPEG","ILSVRC2012_val_00001280.JPEG"
    ],
    "sample103": [
        "ILSVRC2012_val_00043513.JPEG","ILSVRC2012_val_00026564.JPEG","000095642.jpg",
        "ILSVRC2012_val_00002174.JPEG","ILSVRC2012_val_00017250.JPEG","ILSVRC2012_val_00048494.JPEG",
        "10_659_6301763026_67f1036fa9_c.jpg","ILSVRC2012_val_00034742.JPEG","ILSVRC2012_val_00016010.JPEG",
        "ILSVRC2012_val_00022297.JPEG"
    ],
    "sample104": [
        "21_2217_4030096682_e73d0c7f30_c.jpg","26_1071_522549861_97f03a739f_c.jpg",
        "27_1504_6007075536_3750a30773_c.jpg","6_2019_304027208_444e9629a2_c.jpg","16_2766_9232675746_ac49cd269d_c.jpg",
        "20_2056_4618943767_38622a66f5_c.jpg","06692.jpg","21_1466_1371241387_5e6f703cbe_c.jpg",
        "06636.jpg","3_2830_5356853541_6300bc90bd_c.jpg","23_1636_534740799_abff74d94f_c.jpg"
    ],
    "sample105": [
        "ILSVRC2012_val_00011751.JPEG","ILSVRC2012_val_00001651.JPEG","ILSVRC2012_val_00004748.JPEG",
        "ILSVRC2012_val_00011728.JPEG","ILSVRC2012_val_00002988.JPEG","ILSVRC2012_val_00005639.JPEG",
        "ILSVRC2012_val_00012055.JPEG","ILSVRC2012_val_00007137.JPEG","ILSVRC2012_val_00002701.JPEG",
        "ILSVRC2012_val_00000462.JPEG","ILSVRC2012_val_00006271.JPEG"
    ],
    "sample106": [
        "000013869.jpg","volin.jpg","Accordion.jpg","ibanez.jpg","12_2685_2264151502_c406080ebc_c.jpg",
        "17_946_3090741492_34ff65ccd6_c.jpg","17_936_3646317018_553867f082_c.jpg","29_1713_529016050_cd336771a6_c.jpg",
        "000013244.jpg","000013226.jpg"
    ],
    "sample107": [
        "22_634_46924991935_3e3a09bb7a_c.jpg","7_2005_5317851926_8d61274098_c.jpg","ILSVRC2012_val_00018014.JPEG",
        "8_127_357008895_24ba4510da_c.jpg","ILSVRC2012_val_00011038.JPEG","ILSVRC2012_val_00009104.JPEG",
        "ILSVRC2012_val_00011217.JPEG","11_970_1337241238_b29b0f16f8_c.jpg","ILSVRC2012_val_00015250.JPEG",
        "10_1156_6274327525_767b933bfe_c.jpg","ILSVRC2012_val_00015544.JPEG"
    ],
    "sample108": [
        "536736457.jpg","45077563364.jpg","0695327.jpg","000129436.jpg","000166427.jpg",
        "3_1992_3999964691_469855da43_c.jpg","000039849.jpg","000102722.jpg","13_1162_14101489713_8df6f1cb1d_c.jpg",
        "000148987.jpg","A45E60D71B867B9B2101F601C7C7ECBA56EA0BAE_size56_w600_h388.jpg",
        "merlin_200978808_96ee2749-3341-450d-baa9-00efb3347518-master1050.jpg",
        "20210310002013.jpg","40E7FC87F074701A98053D733DAE15AE3B11F5E7_size64_w1440_h810.jpg",
        "6597dbfbe4b00c772112517c_m.png","2-200106221949244.jpg",
        "Lockheed_Martin_F-22A_Raptor_JSOH.jpg","64300b8ae4b020d074b648931.jpg"
    ],
    "sample109": [
        "22_133_26405848950_44c6e31373_c.jpg","27_232_49009045008_4d8b924672_c.jpg","3_113_5760123181_2fd356ceaa_c.jpg",
        "22_680_33987634432_9d27fef794_c.jpg","3_220_43884793680_a9baf6caa7_c.jpg","21_334_3172067597_eb0cca5f59_c.jpg",
        "19_498_9551900266_7694d5a921_c.jpg","tyrannosaur.jpg","10_208_5349690084_77447dc909_c.jpg",
        "3_498_8595235915_f5015049de_c.jpg","panda.jpg","unnamed.png","tiger.jpg","d2e1db6b963fe494.jpg",
        "5bc965fba310eff36901a88f.jpg","c0522c9eb0834fafa2a9f0fe39d27db1.jpg","2019030834861153.jpg",
        "2ab2903b15c2451680a95402aa58b93b.jpg","7ba16d4343e8420ba0828b510fe9b1fc.jpg","2104-29f41083-f134-62e2-ecbc-710947cd94be.png"
    ]
}

# 获取 CSV 文件列表
csv_files = sorted(glob(os.path.join(csv_dir, "Qwen2-VL-7B-Instruct_sample*.csv")))

for csv_path in csv_files:
    filename = os.path.basename(csv_path)
    sample_name = filename.replace("Qwen2-VL-7B-Instruct_", "").replace(".csv", "")
    print(f"Processing {sample_name}...")

    target_dir = os.path.join(pics_dir, sample_name)
    os.makedirs(target_dir, exist_ok=True)

    # 先处理 CSV 中的图片
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_files = row["image_files"].strip()
            if not image_files:
                continue
            for img_path in image_files.split(";"):
                img_name = os.path.basename(img_path.strip())
                src = os.path.join(pics_dir, img_name)
                dst = os.path.join(target_dir, img_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # 再处理 extra_images（100~109）
    if sample_name in extra_images:
        for img_name in extra_images[sample_name]:
            src = os.path.join(pics_dir, img_name)
            dst = os.path.join(target_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
