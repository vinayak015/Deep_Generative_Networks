from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "pre_load": "latest",
        "tokenizer_file": f"tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "n_heads": 8,
        "num_layers": 6,
    }


def get_weight_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_*"
    weight_files = list(Path(model_folder).glob(model_filename))
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return str(weight_files[-1])
