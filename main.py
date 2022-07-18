import json, click, time
from pathlib import Path

from train_v2 import train_core
from utils import check_dir, get_items_from_file, PathlibEncoder
from sklearn.model_selection import train_test_split, KFold 
from datasets import CLASSIFICATION_DATASETS

@click.command('train')
@click.option("param-path", type=click.Path(), default=Path.home()/"Code"/"ExPN-Net"/"param.list")
def train(param_path):
    confs = get_items_from_file(param_path, format='json')
    random_seed = 42
    
    if confs['dimensions'] == '2':
        save_dir_name = f"{time.strftime('%m%d_%H%M')}-{confs['net']}-slice_{confs['in_channels']}-lr_{confs['lr']}-{confs['loss_name']}-{confs['optim']}{confs['postfix']}"
    else:
        save_dir_name = f"{time.strftime('%m%d_%H%M')}-{confs['net']}-lr_{confs['lr']}-{confs['loss_name']}-{confs['optim']}{confs['postfix']}"
    confs['out_dir'] = check_dir(f"/homes/yliu/Data/pn_cls_exp/{confs['dataset_name']}/{save_dir_name}")
    with open(confs['out_dir']/'param.list', 'w') as f:
        json.dump(confs, f, indent=2, cls=PathlibEncoder)
    
    dataset_type = CLASSIFICATION_DATASETS[f"{confs['dimensions']}D"][confs['dataset_name']]['FN']
    dataset_list = CLASSIFICATION_DATASETS[f"{confs['dimensions']}D"][confs['dataset_name']]['PATH']
    datalist = get_items_from_file(dataset_list, format='json')

    files_train, files_valid = train_test_split(
        datalist, test_size=confs['test_split_ratio'], random_state=random_seed
    )
    train_core(files_train, files_valid, dataset_type, **confs)


if __name__ == '__main__':
    train()