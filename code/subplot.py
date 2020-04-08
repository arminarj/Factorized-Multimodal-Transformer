import os

import numpy as np
import matplotlib.pyplot as plt
import h5py

plt_type_conf = {'update':{}, 'grad':{}}
# conf = 'conf_clip_cuda0_152_dp_0E+00_lr_1E-04_grad_5E-03'
# epochs = [1]#, 10, 50, 100, 150, 200, 500]
# epochs = [2, 11, 51, 101, 151, 201, 499]
# n_epochs = len(epochs)
modules = ['gru'] + ['transformer_encoder.layers.%d.' % i for i in range(7, -1, -1)]
n_modules = len(modules)
config_dir = '/home/chengfem/Heirarchical_Transformer/local_search_configs'
h5_dir = '/work/chengfem/model/'
n = 0
model_prefix = 'sgd'
for config_file in os.listdir(config_dir):
    config_name = config_file.split('.')[0]
    if not config_name.startswith('conf_b0_cuda'):
        continue
    plt_type_conf['update'][config_name] = []
    plt_type_conf['grad'][config_name] = []


for h5_file in os.listdir(h5_dir):
    if not h5_file.endswith('.hdf5'):
        continue
    # print('h5 file: %s' % h5_file)
    h5_name = h5_file.split('.')[0]
    h5_parts = h5_name.split('_')
    if model_prefix != h5_parts[0]:
        continue
    type_idx = 1
    plt_type = h5_parts[type_idx]
    if plt_type not in ['update', 'grad']:
        continue
    epoch = int(h5_parts[-1])
    conf = '_'.join(h5_parts[type_idx+1:-1])
    if conf not in plt_type_conf[plt_type]:
        continue
    # print('conf %s' % conf)
    plt_type_conf[plt_type][conf].append(epoch)


for plt_type in ['update', 'grad']:
    for conf, orig_epochs in plt_type_conf[plt_type].items():
        epochs = sorted(orig_epochs)
        n_epochs = len(orig_epochs)
        if n_epochs == 0:
            continue
        # print('fig size %d x %d' % (n_epochs, len(modules)))
        f = plt.figure(figsize=(4*n_epochs, 12))
        # f, axes = plt.subplots(nrows=n_modules, ncols=n_epochs, figsize=(5*n_epochs, 10))
        f.tight_layout()
        for j, ep in enumerate(epochs):
            hf = h5py.File(h5_dir + '%s_%s_%s_%d.hdf5' % (model_prefix, plt_type, conf, ep), 'r')
            row = 0
            for mod_name in modules:
                x = []
                for k, v in hf.items():
                    if k.startswith(mod_name):
                        x.append(np.array(v).flatten())
                if len(x) == 0:
                    continue
                num_bins = 30
                # print('plotting epochs[%d]=%d %s at %d' % (j, epochs[j], mod_name, j + row * n_epochs))
                ax = f.add_subplot(n_modules, n_epochs, j + 1 + row * n_epochs)
                n, bins, patches = ax.hist(np.concatenate(x), num_bins, facecolor='blue', alpha=0.5)
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                # if plt_type == 'update':
                #     max_x = 0.0006
                #     plt.xlim((0, max_x))
                # else:
                #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                if row == 0:
                    plt.title('Epoch %d \n%s' % (ep, mod_name))
                else:
                    plt.title(mod_name)
                row += 1
                plt.yscale('log')
            hf.close()

        plt.subplots_adjust(hspace = 1.0)
        plt.savefig('%s_%s_subplots.png' % (plt_type, conf), dpi=200)
        print("saved %s_%s_subplots.png" % (plt_type, conf))