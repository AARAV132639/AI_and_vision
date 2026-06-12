import os
def check_labels(base_path):
    labels = set()

    for img_name in os.listdir(base_path):

        label = img_name.split('_')[0].split('(')[0]
        labels.add(label)

    print(f"Unique mudras found:{labels}")

check_labels("data/raw/images/train")