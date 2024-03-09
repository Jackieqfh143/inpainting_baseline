import glob
import os
from random import shuffle


if __name__ == '__main__':
    train_list = glob.glob(os.path.join('/home/codeoops/CV/data/Paris/train','*.jpg'))
    val_list = glob.glob(os.path.join('/home/codeoops/CV/data/Paris/eval','*.png'))

    print('Total training imgs: ',len(train_list))
    print('Total val imgs: ',len(val_list))

    shuffle(train_list)
    shuffle(val_list)

    save_dir = './data_flist/paris'
    with open(save_dir+'/train.flist','w') as f:
        for f_name in train_list:
            print('writing {}'.format(f_name))
            f.write(f_name+'\n')

    with open(save_dir+'/val.flist','w') as f:
        for f_name in val_list:
            print('writing {}'.format(f_name))
            f.write(f_name+'\n')

    print('Writing Finished!')