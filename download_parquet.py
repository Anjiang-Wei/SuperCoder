from datasets import load_dataset
import fire 

def main(ds_path = 'random1123anonymized/supercoder'):
    ds = load_dataset(ds_path, split='train')
    sv_path = ds_path.split('/')[-1]
    ds.to_parquet(f'{sv_path}_train.parquet')

    ds = load_dataset(ds_path, split='val')
    ds.to_parquet(f'{sv_path}_val.parquet')

if __name__ == '__main__':
    fire.Fire(main)