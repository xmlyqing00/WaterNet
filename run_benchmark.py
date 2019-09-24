import os

root_folder = '/Ship01/Dataset/water/'
full_videos = ['stream0', 'stream1', 'stream3_small', 'stream4', 'boston_harbor2_small_rois', 'buffalo0_small', 'canal0']

sample_flag = ' --sample '
no_AA_flag = '--no-AA'

def run_benchmark():

    with open(os.path.join(root_folder, 'eval.txt'), 'r') as f:
        tmp = f.readlines()
        sequences_names = [x.strip() for x in tmp]
    
    try:
        sequences_names.remove('')
    except ValueError as e:
        pass

    print('Seq names:', sequences_names)

    for seq in sequences_names:
        cmd = f'python3 eval_AANet.py -c=models/cp_AANet_199.pth.tar -v {seq}'
        if seq not in full_videos:
            cmd += sample_flag
        os.system(cmd)
    
    cmd = 'python3 /Ship01/Sources/VOS-evaluation/eval_waterdataset.py --method AANet_segs --update'
    os.system(cmd)


if __name__ == '__main__':
    run_benchmark()