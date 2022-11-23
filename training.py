import argparse
import os
import shutil

import yaml

import matplotlib.pyplot as plt
import dataset
from tqdm import tqdm
import torch
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config


cfg = load_config("config.yaml")

loader_dict = {
    'MyLoader': dataset.LEVIRLoader
}


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default=cfg['paths']['datapath']
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="log path",
        default=cfg['paths']['logpath']
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.logpath):
        os.mkdir(parsed_arguments.logpath)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.logpath)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.logpath = os.path.join(
        parsed_arguments.logpath, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    criterion,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    save_after,
    device,
    patience_limit
):

    model = model.to(device)

    tool4metric = ConfuseMatrixMeter(n_class=2)

    training_loss_list = []
    validation_loss_list = []
    epcs = []
    patience = 0

    def evaluate(reference, testimg, mask):
        # All the tensors on the device:
        reference = reference.to(device).float()
        testimg = testimg.to(device).float()
        mask = mask.to(device).float()

        # Evaluating the model:
        generated_mask = model(reference, testimg).squeeze(1)

        # Loss gradient descend step:
        it_loss = criterion(generated_mask, mask)

        # Feeding the comparison metric tool:
        bin_genmask = (generated_mask.to("cpu") >
                       0.5).detach().numpy().astype(int)
        mask = mask.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=bin_genmask, gt=mask)

        return it_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = cfg['params']['epoch_loss']
        for (reference, testimg), mask in tqdm(dataset_train):
            # Reset the gradients:
            optimizer.zero_grad()

            # Loss gradient descend step:
            it_loss = evaluate(reference, testimg, mask)
            it_loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        epcs.append(epc)
        training_loss_list.append(epoch_loss)

        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()
        writer.flush()

        ### Save the model ###
        if epc % save_after == 0:
            torch.save(
                model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc))
            )

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        with torch.no_grad():
            for (reference, testimg), mask in dataset_val:
                epoch_loss_eval += evaluate(reference,
                                            testimg, mask).to("cpu").numpy()

        epoch_loss_eval /= len(dataset_val)

        validation_loss_list.append(epoch_loss_eval)

        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU_val class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()

    def early_stop(epc):
        nonlocal patience
        if validation_loss_list[epc] > min(validation_loss_list):
            patience += 1
            if patience >= patience_limit:
                return True
            else:
                patience = 0

    for epc in range(epochs):
        training_phase(epc)
        validation_phase(epc)
        if early_stop(epc):
            break
        # scheduler step
        scheduler.step()
        

    
    plt.plot(epcs, training_loss_list, color = "red", label='Training loss')
    plt.plot(epcs, validation_loss_list, color = "blue", label='Evaluation loss')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.savefig('dataset/results/loss/loss_graph.png')


def run():
    if cfg['settings']['set_seed'] == True:
        # set the random seed
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.logpath)

    # Inizialitazion of dataset and dataloader:
    trainingdata = loader_dict['MyLoader'](args.datapath, "train")
    validationdata = loader_dict['MyLoader'](args.datapath, "val")
    data_loader_training = DataLoader(
        trainingdata, batch_size=cfg["params"]["batch_size"], shuffle=True)
    data_loader_val = DataLoader(
        validationdata, batch_size=cfg["params"]["batch_size"], shuffle=True)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model()
    restart_from_checkpoint = False
    model_path = None
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    # define the loss function for the model training.
    criterion = torch.nn.BCELoss()

    # choose the optimizer in view of the used dataset
    # if cfg['params']['dataset'] == 'LEVIR-CD':
    #     # Optimizer with tuned parameters for LEVIR-CD
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=0.00356799066427741,
    #                                   weight_decay=0.009449677083344786, amsgrad=False)
    # elif cfg['params']['dataset'] == 'WHU-CD':
    #     # Optimizer with tuned parameters for WHU-CD
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=0.002596776436816101,
    #                                   weight_decay=0.008620171028843307, amsgrad=False)

    # generalize
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['params']['lr'],
                                  weight_decay=cfg['params']['weight_decay'], amsgrad=False)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.logpath, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.logpath,
        writer,
        epochs=cfg["params"]["epochs"],
        save_after=1,
        device=device,
        patience_limit = cfg["params"]["patience_limit"]
    )
    writer.close()


if __name__ == "__main__":
    run()
