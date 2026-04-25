
import argparse
import random
import sys
import time

import builder
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from trainer import trainer
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
import os, json
import pickle
from ast import literal_eval


def _set_by_dotpath(root, key, value):
    parts = key.split(".")
    cur = root
    for p in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(p, {})
        else:
            if not hasattr(cur, p) or getattr(cur, p) is None:
                setattr(cur, p, {})
            cur = getattr(cur, p)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)

def update_from_kv_inplace(configs_obj, kv_list):
    for kv in (kv_list or []):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        # type transformation: int/float/bool/None/list/dict etc
        try:
            v_cast = literal_eval(v)
        except Exception:
            low = v.lower()
            if low in ("true", "false"):
                v_cast = (low == "true")
            elif low in ("none", "null", "~"):
                v_cast = None
            else:
                v_cast = v
        _set_by_dotpath(configs_obj, k, v_cast)


def main() -> None:
    # seed = 233
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    configs.evalmode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["pretrain", "finetune", "test"], default="pretrain",
                    help="pretrain whole model or finetune the head only")
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path to load for finetune")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="freeze encoder when finetuning (default True in finetune mode)")
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--enc_lr", type=float, default=2e-5)   # If the last layer needs to be unfrozen
    parser.add_argument("--finetune_epochs", type=int, default=5)

    parser.add_argument("--load", action="store_true", help="config file")
    parser.add_argument("--special_node_with_grad", action="store_true", help="config file")
    parser.add_argument("--train_size", default=0, type=int, help="config file")
    parser.add_argument("--table_by_cirid_file", type=str, default="", help="table_by_cirid")

    parser.add_argument("--backend", type=str, default="", help="backend")
    parser.add_argument("--cut", type=str, default=None, help="cut")
    parser.add_argument("--extra_data", type=str, default=None, help="extra_data")


    configs.load("config.yaml", recursive=True)

    args, opts = parser.parse_known_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ① Inject recognized parameters
    configs.update(vars(args))

    # ② Continue processing extra parameters in key=value form (if any)
    #    Here we assume configs essentially holds a dict, 
    #    which can be accessed or updated as needed
    update_from_kv_inplace(configs, opts)

    if configs.device == "gpu":
        device = torch.device("cuda")
    elif configs.device == "cpu":
        device = torch.device("cpu")

    with open(configs.table_by_cirid_file, "rb") as f:
        table_by_cirid = pickle.load(f)

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    args.run_dir = "/tmp"

    logger.info(" ".join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + "\n" + f"{configs}")

    model = builder.make_model()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Size: {total_params}")
    dataflow = {}
    if not args.load:
        dataset = builder.make_dataset()
        for split in ["train", "valid", "test"]:
        # for split in ["train", "valid"]:
            dataflow[split] = DataLoader(
                dataset.get_data(device, split), batch_size=configs.batch_size
            )

    criterion = builder.make_criterion()
    # optimizer = builder.make_optimizer(model)
    # scheduler = builder.make_scheduler(optimizer)

    # ======= pretrain or finetune =======
    if args.mode == "finetune":
        assert args.ckpt, "finetune requires --ckpt"
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        model.load_state_dict(state, strict=False)

        # By default, only fine-tune the head: freeze encoder parameters
        if args.freeze_encoder or True:
            if hasattr(model, "encoder"):
                for p in model.encoder.parameters():
                    p.requires_grad = False
                model.encoder.eval()

        # Optimize head only; to unfreeze the last layer, add it to param_groups
        enc_params = list(model.encoder.parameters())
        head_params = list(model.head.parameters())
        param_groups = [
            {"params": enc_params,  "lr": args.enc_lr},   
            {"params": head_params, "lr": args.head_lr},  
        ]
        # optional：
        # if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        #     for p in model.encoder.layers[-1].parameters():
        #         p.requires_grad = True
        #     param_groups.append({"params": model.encoder.layers[-1].parameters(), "lr": args.enc_lr})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=getattr(configs.optimizer, "weight_decay", 0.0),
        )
        scheduler = builder.make_scheduler(optimizer)
        configs.num_epochs = args.finetune_epochs

        # # train the entire model（encoder + head）
        # for p in model.parameters():
        #     p.requires_grad = True

        # enc_params = list(model.encoder.parameters())
        # head_params = list(model.head.parameters())
        # param_groups = [
        #     {"params": enc_params,  "lr": args.enc_lr},   
        #     {"params": head_params, "lr": args.head_lr},  
        # ]
        # optimizer = torch.optim.AdamW(
        #     param_groups,
        #     weight_decay=getattr(configs.optimizer, "weight_decay", 0.0),
        # )
        # scheduler = builder.make_scheduler(optimizer)
        # configs.num_epochs = args.finetune_epochs

    elif args.mode == "pretrain":
        # Pretraining: optimize the entire model
        optimizer = builder.make_optimizer(model)
        scheduler = builder.make_scheduler(optimizer)
    
    elif args.mode == "test":
        assert args.ckpt, "test require --ckpt"
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        model.load_state_dict(state, strict=False)
        optimizer, scheduler = None, None


    print(criterion)
    my_trainer = trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataflow,
        backend=args.backend,
        mode=args.mode,
        seed=args.seed,
        cut=args.cut,
        extra_data=args.extra_data,
        table_by_cirid=table_by_cirid,
        xobs_std=None,
        default_shots=1024
    )
    
    if args.mode in ["pretrain", "finetune"]:
        training_time = -1
        training_start_time = time.time()
        my_trainer.train()
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        print(f"Training time: {training_time:.2f}s")

    if args.mode == "test":
        test_time = -1
        test_start_time = time.time()
        my_trainer.test()
        test_end_time = time.time()
        
        test_time = test_end_time - test_start_time
        
        print(f"Testing time: {test_time:.2f}s")
        my_trainer.saveall()


if __name__ == "__main__":
    main()
