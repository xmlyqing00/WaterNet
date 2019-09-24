import os
import os

root_folder = '/Ship01/Dataset/water'
with open(os.path.join(root_folder, 'tmp'), 'r') as f:
    replace_rules = f.readlines()

target_folder = 'results/RGMP/stream1'

for rule in replace_rules:
    pair = rule.split()
    if len(pair) == 0:
        break

    ori_path = os.path.join(root_folder, target_folder, pair[0].zfill(5) + '.png')
    dst_path = os.path.join(root_folder, target_folder, pair[1] + '.png')

    # print(ori_path, dst_path)

    cmd = f'mv {ori_path} {dst_path}'
    os.system(cmd)
    
