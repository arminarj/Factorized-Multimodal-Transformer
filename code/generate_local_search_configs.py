import functools
import json
import operator
import os
import random
from collections import OrderedDict

random.seed(0)
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=False, type=int, default=4)
    args = vars(ap.parse_args())
    val_options = {
        # "max_grad": [0.005], 0.0001],
        "lr": [0.0001, 0.00001],
        # "seed": [0, 1, 2], #], 3, 4]
        "dropout": [0.2, 0.5], #, 0.8], #, 0.6, 0.8],
        "gru_lr": [0.0001, 0.00001], #, 0.0001, 0.00001],
        "gru_dropout": [0.2] #, 0.5, 0.8]
    }
    option_lens = [len(options) for options in val_options.values()]
    num_comb = functools.reduce(operator.mul, option_lens, 1)
    gpus = int(args["gpus"])
    best_config_dir = '../best_2_configs'
    n = 0
    for config_file in os.listdir(best_config_dir):
        config = json.load(open(os.path.join(best_config_dir, config_file)), object_pairs_hook=OrderedDict)
        config_num = config_file.split('.')[0].split('_')[-1]
        for i in range(num_comb):
            cuda = n % gpus
            n += 1
            new_config = config.copy()
            new_config['cuda'] = cuda
            new_config['epoch_num'] = 1000
            new_config['seed'] = 0
            same = True
            for key, options in val_options.items():
                idx = i % len(options)
                same &= (key in new_config and new_config[key] == options[idx])
                new_config[key] = options[idx]
                i -= idx
                if i >= len(options):
                    i //= len(options)
            # if same:
            #     continue
            file_name = '../local_search_configs/' + 'conf_a1_cuda%d_%s_dp_%.0E_lr_%.0E_gru_lr_%.0E.json' % (
            cuda, config_num, new_config['dropout'], new_config['lr'], new_config['gru_lr'])
            with open(file_name, 'w') as outfile:
                json.dump(new_config, outfile)
