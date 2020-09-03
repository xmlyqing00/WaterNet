import os
import argparse

root_folder = '/Ship01/Dataset/water_v2/'
full_videos = ['stream0', 'stream1', 'stream3_small', 'stream2', 'boston_harbor2_small_rois', 'buffalo0_small', 'canal0', 'mexico_beach_clip0', 'holiday_inn_clip0', 'gulf_crest', 'pineapple_willy_clip0', 'pineapple_willy_clip1']

test_videos = full_videos
list_name = 'val.txt' # 'eval_all.txt'

def get_sequence_list():
    
    with open(os.path.join(root_folder, list_name), 'r') as f:
        tmp = f.readlines()
        sequence_list = [x.strip() for x in tmp]
    
    try:
        sequence_list.remove('')
    except ValueError as e:
        pass

    print('Seq names:', sequence_list)
    return sequence_list


def get_scores(method_name):
    method_str = f' --method {method_name}'
    base_cmd = 'python3 /Ship01/SourcesArchives/VOS-evaluation/eval_waterdataset.py --update'
    cmd = base_cmd + method_str
    os.system(cmd)

def eval_WaterNet(args):

    
    sequence_list = get_sequence_list()
    base_cmd = 'python3 eval_WaterNet.py -c=models/cp_WaterNet_199.pth.tar'

    method_name = 'WaterNet'
    if args.no_aa:
        base_cmd += ' --no-aa'
        method_name += '_no_aa'
    if args.no_conf:
        base_cmd += ' --no-conf'
        method_name += '_no_conf'

    if not args.score:
        for seq in sequence_list:
            cmd = base_cmd + f' -v {seq}'
            if seq not in test_videos:
                cmd += ' --sample '
            os.system(cmd)
    
    # get_scores(method_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bencharmark')
    parser.add_argument(
        '--method', default='WaterNet', type=str,
        help='Input the method name (default: WaterNet).')
    parser.add_argument(
        '--score', action='store_true',
        help='Compute the scores without re-run the benchmark.')
    parser.add_argument(
        '--no-conf', action='store_true', 
        help='For WaterNet (default: none).')
    parser.add_argument(
        '--no-aa', action='store_true',
        help='For WaterNet (default: none).')
    parser.add_argument(
        '--no-online', action='store_true',
        help='For OSVOS and MSK (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    if args.method == 'WaterNet':
        eval_WaterNet(args)

