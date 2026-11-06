import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import json
from utils.comparison import load_class_order, forgetting


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    order_file = args.get("class_order_file")
    order_indices = args.get("class_order_idx", [0, 1, 2]) if order_file else [None]
    if isinstance(order_indices, int):
        order_indices = [order_indices]
    if not order_indices or len(set(order_indices)) != len(order_indices):
        raise ValueError("class_order_idx must contain distinct order indices")
    for order_idx in order_indices:
        order = load_class_order(order_file, order_idx) if order_file else None
        for seed in seed_list:
            run_args = copy.deepcopy(args)
            run_args["seed"] = seed
            run_args["device"] = copy.deepcopy(device)
            run_args["class_order"] = order
            run_args["class_order_idx"] = order_idx
            _train(run_args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], args["seed"],args["convnet_type"],)
    if args.get("class_order_idx") is not None:
        logfilename += "_order{}".format(args["class_order_idx"])
    logging.basicConfig(force=True,level=logging.INFO,format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args["dataset"],args["shuffle"],args["seed"],args["init_cls"],args["increment"], class_order=args.get("class_order"))
    model = factory.get_model(args["model_name"], args)
    model.save_dir=logs_name

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}

    acc_history = []
    with open(logfilename + ".metrics.jsonl", "w") as stream:
        stream.write(json.dumps({"seed": args["seed"],
                                 "class_order_idx": args.get("class_order_idx"),
                                 "class_order": data_manager._class_order}) + "\n")

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        # cnn_accy, nme_accy = model.eval_task()
        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        model.after_task()
        acc_history.append(cnn_accy["acc_per_task"])
        fg, seca_fg = forgetting(acc_history)
        record = {"task": task, "acc": float(cnn_accy["top1"]),
                  "acc_per_task": acc_history[-1], "fg": fg, "seca_fg": seca_fg}
        with open(logfilename + ".metrics.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
        logging.info("FG: %.6f; SECA FG (continuum 1.2.7): %.6f", fg, seca_fg)

       
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
