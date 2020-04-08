import json
import os
import random
from collections import OrderedDict

random.seed(0)
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=False, type=int, default=0)
    ap.add_argument("--end", required=False, type=int, default=64)
    ap.add_argument("--gpus", required=False, type=int, default=2)
    args = vars(ap.parse_args())
    config = {"cuda": 2, "lr": 0.001, "dropout": 0.2, "n_head": 4, "proj_dim_a": 20,
              "proj_dim_v": 40, "conv_dims": [40, 40], "ff_dim_final": 256, "n_layers": 6, "dim_total_proj": 512}
    val_options = {
        "lr": [0.0012, 0.0011, 0.0009, 0.0008],

        "dropout": [0.1, 0.15, 0.18, 0.22, 0.25, 0.3]
    }
    option_lens = [len(options) for options in val_options.values()]
    num_comb = sum(option_lens)
    start = int(args["start"])
    end = int(args["end"])
    gpus = int(args["gpus"])
    best_config_dir = '../best_configs'
    for config_file in os.listdir(best_config_dir):
        config = json.load(open(os.path.join(best_config_dir, config_file)), object_pairs_hook=OrderedDict)
        config_num = config_file.split('.')[0].split('_')[-1]
        for i in range(num_comb):
            cuda = i % gpus
            file_name = '../local_search_configs/' + 'config_cuda%d_%s_%d.json' % (cuda, config_num, i + start)
            new_config = config.copy()
            new_config['cuda'] = cuda
            same = False
            for key, options in val_options.items():
                if i >= len(options):
                    i -= len(options)
                    continue
                same = (new_config[key] == options[i])
                new_config[key] = options[i]
                break
            if same:
                continue
            with open(file_name, 'w') as outfile:
                json.dump(new_config, outfile)
