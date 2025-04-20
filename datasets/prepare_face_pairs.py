import random
import os
from collections import defaultdict

base_path = './AdvCelebA'
attack_status_file = os.path.join(base_path, 'attack_CelebA.txt')
identity_file = os.path.join(base_path, 'identity_CelebA.txt')
attack_map = {}
with open(attack_status_file, "r") as f:
    for line in f:
        fname, is_attacked = line.strip().split()
        attack_map[fname] = int(is_attacked)

identity_map = {}
with open(identity_file, 'r') as f:
    for line in f:
        fname, identity = line.strip().split()
        if fname in attack_map and attack_map[fname] == 0:
            identity_map[fname] = int(identity)

identity_to_images = defaultdict(list)
for fname, identity in identity_map.items():
    identity_to_images[identity].append(fname)

eligible_identities = [id_ for id_, imgs in identity_to_images.items() if len(imgs) >= 2]

positive_pairs = []
for _ in range(500):
    identity = random.choice(eligible_identities)
    imgs = random.sample(identity_to_images[identity], 2)
    positive_pairs.append((imgs[0], imgs[1], 1))

all_identities = list(identity_to_images.keys())
negative_pairs = []
for _ in range(500):
    id1, id2 = random.sample(all_identities, 2)
    img1 = random.choice(identity_to_images[id1])
    img2 = random.choice(identity_to_images[id2])
    negative_pairs.append((img1, img2, 0))

final_pairs = positive_pairs + negative_pairs
random.shuffle(final_pairs)

with open("clean_eval_pairs.txt", "w") as f:
    for img1, img2, label in final_pairs:
        f.write(f"{img1} {img2} {label}\n")

