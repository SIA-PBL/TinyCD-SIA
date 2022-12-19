import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# clear cache
os.system('python clear_cache.py')

import argparse
import shutil

import yaml

import matplotlib.pyplot as plt
from dataset import *
from tqdm import tqdm
import torch
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from utils.pytorchtools import EarlyStopping

import albumentations as alb
from albumentations.pytorch import ToTensorV2

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config

cfg = load_config("config.yaml")

loader_dict = {
    # 'MyLoader': dataset.SPN7Loader
    'spacenet7': spacenet7.SPN7Loader
}

resize_func     = alb.Compose([
    alb.Resize(cfg['data_config']['patch_size'], cfg['data_config']['patch_size']),
])

array2tens_func = alb.Compose([
    ToTensorV2
])
    

P_SIZE = cfg['data_config']['patch_size']
OV_SIZE = cfg['data_config']['overlap_size']

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
    patience_limit):

    model = model.to(device)
    tool4metric = ConfuseMatrixMeter(n_class=2)

    training_loss_list      = []
    validation_loss_list    = []
    epcs                    = []

    early_stopping = EarlyStopping(patience=patience_limit, verbose=True)
    
    segl_weight = cfg['loss']['seg_weight']
    cdl_weight  = cfg['loss']['cd_weight']  
    
    best_loss = 0.0

    def train_evaluate(preimg, postimg, preseg, postseg, cd_gt):
        preimg = preimg.to(device).float()
        posimg = posimg.to(device).float()
        preseg = preseg.to(device).float()
        posseg = posseg.to(device).float()
        cd_gt = cd_gt.to(device).float()
        
        # Evaluating the model
        p_cd, p_cd_r, p_preseg, p_posseg = model(preimg, posimg)
        p_cd    = p_cd.squeeze()
        p_cd_r  = p_cd_r.squeeze()
        p_preseg = p_preseg.squeeze()
        p_posseg = p_posseg.squeeze()
        
        # Compute Loss 
        it_cd_loss = criterion(p_cd, cd_gt) + criterion(p_cd_r, cd_gt)
        it_seg_loss = criterion(p_preseg, preseg) + criterion(p_posseg, posseg)
        
        # Feeding the comparison metric tool:
        cd_pr = (p_cd.to('cpu') > 0.5).detach().numpy().astype(int)
        cd_gt = cd_gt.to('cpu').numpy().astype(int)
        tool4metric.update_cm(pr=cd_pr, gt=cd_gt)

        return it_cd_loss, it_seg_loss

    def test_evaluate(prepatches, pospatches, preseg, posseg, cd_gt, ov_guidance):
        # All the tensors on the device:
        #preimg = preimg.to(device).float()
        #posimg = posimg.to(device).float()
        preseg = preseg.to(device).float()
        posseg = posseg.to(device).float()
        cd_gt = cd_gt.to(device).float()
        
        cd_p_frame      = np.zeros(np.shape(cd_gt))
        cd_p_r_frame    = np.zeros(np.shape(cd_gt))
        preseg_frame    = np.zeros(np.shape(preseg))
        posseg_frame    = np.zeros(np.shape(posseg))

        # Evaluating the model:
        for p_idx in range(len(prepatches)):
            cur_pre_patch = prepatches[p_idx].to(device).float()
            cur_pos_patch = pospatches[p_idx].to(device).float()

            cur_cd, cur_cd_reverse, cur_preseg, cur_posseg = model(cur_pre_patch, cur_pos_patch)
            cur_cd              = resize_func(image=cur_cd.squeeze().type(torch.uint8).cpu().numpy())
            cur_cd              = cur_cd['image']
            cur_cd_reverse      = resize_func(image=cur_cd_reverse.squeeze().type(torch.uint8).cpu().numpy())
            cur_cd_reverse      = cur_cd_reverse['image']
            cur_preseg          = resize_func(image=cur_preseg.squeeze().type(torch.uint8).cpu().numpy())
            cur_preseg          = cur_preseg['image']
            cur_posseg          = resize_func(image=cur_posseg.squeeze().type(torch.uint8).cpu().numpy())
            cur_posseg          = cur_posseg['image']
            
            h_num = int(math.sqrt(len(prepatches)))
            cur_h_idx = int(p_idx/h_num)
            cur_w_idx = int(p_idx%h_num)

            cd_p_frame[cur_h_idx * P_SIZE - cur_h_idx * OV_SIZE : (cur_h_idx+1) * P_SIZE - cur_h_idx * OV_SIZE,
                       cur_w_idx * P_SIZE - cur_w_idx * OV_SIZE : (cur_w_idx+1) * P_SIZE - cur_w_idx * OV_SIZE] += cur_cd
            cd_p_r_frame[cur_h_idx * P_SIZE - cur_h_idx * OV_SIZE : (cur_h_idx+1) * P_SIZE - cur_h_idx * OV_SIZE,
                         cur_w_idx * P_SIZE - cur_w_idx * OV_SIZE : (cur_w_idx+1) * P_SIZE - cur_w_idx * OV_SIZE] += cur_cd_reverse
            preseg_frame[cur_h_idx * P_SIZE - cur_h_idx * OV_SIZE : (cur_h_idx+1) * P_SIZE - cur_h_idx * OV_SIZE,
                         cur_w_idx * P_SIZE - cur_w_idx * OV_SIZE : (cur_w_idx+1) * P_SIZE - cur_w_idx * OV_SIZE] += cur_preseg
            posseg_frame[cur_h_idx * P_SIZE - cur_h_idx * OV_SIZE : (cur_h_idx+1) * P_SIZE - cur_h_idx * OV_SIZE,
                         cur_w_idx * P_SIZE - cur_w_idx * OV_SIZE : (cur_w_idx+1) * P_SIZE - cur_w_idx * OV_SIZE] += cur_posseg

        cd_p_frame /= ov_guidance
        cd_p_frame = array2tens_func(image=cd_p_frame)
        cd_p_frame = cd_p_frame['image'].to(device).float()
        
        cd_p_r_frame /= ov_guidance
        cd_p_r_frame = array2tens_func(image=cd_p_r_frame)
        cd_p_r_frame = cd_p_r_frame['image'].to(device).float()
        
        preseg_frame /= ov_guidance
        preseg_frame = array2tens_func(image=preseg_frame)
        preseg_frame = preseg_frame['image'].to(device).float()

        posseg_frame /= ov_guidance
        posseg_frame = array2tens_func(image=posseg_frame)
        posseg_frame = posseg_frame['image'].to(device).float()

        # Loss gradient descend step:
        it_cd_loss = criterion(cd_p_frame, cd_gt) + criterion(cd_p_r_frame, cd_gt)
        it_seg_loss = criterion(preseg_frame, preseg) + criterion(posseg_frame, posseg)

        # Feeding the comparison metric tool:
        cd_pr           = (cd_p_frame.to("cpu") > 0.5).detach().numpy().astype(int)
        cd_gt           = cd_gt.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=cd_pr, gt=cd_gt)

        return it_cd_loss, it_seg_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        
        model.train()
        epoch_loss  = cfg['params']['epoch_loss']
        
        for sample in tqdm(dataset_train):
            # Data setting
            preimg = sample['pre']
            posimg = sample['pos']
            staimg = sample['sta']
            preseg = sample['pregt']
            posseg = sample['posgt']
            cd_gt  = sample['cdgt']

            # Reset the gradients:
            optimizer.zero_grad()

            # Loss gradient descend step:
            it_cd_loss, it_seg_loss = train_evaluate(preimg, posimg, preseg, posseg, cd_gt)
            
            it_loss = cdl_weight * it_cd_loss + segl_weight * it_seg_loss
            it_loss.backward()
            optimizer.step()
            
            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        epcs.append(epc)
        training_loss_list.append(epoch_loss)

        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU class change/epoch", scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch", scores_dictionary["F1_1"], epc)
        
        print("IoU class change for epoch {} is {}".format(epc, scores_dictionary["iou_1"]))
        print("F1 class change for epoch {} is {}".format(epc, scores_dictionary["F1_1"]))
        writer.flush()

        ### Save the model ###
        if epc % save_after == 0:
            torch.save(model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc)))

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        
        with torch.no_grad():
            for sample  in dataset_val:
                # Data setting
                prepatches = sample['pre']
                pospatches = sample['pos']
                stapatches = sample['sta']
                preseg = sample['pregt']
                posseg = sample['posgt']
                cd_gt  = sample['cdgt']
                pre_name = sample['prename']
                pos_name = sample['posname']
                
                ov_guidance = sample['ov_g']

                it_cd_loss, it_seg_loss = test_evaluate(prepatches, pospatches, preseg, posseg, cd_gt, ov_guidance)
                
                it_loss = cdl_weight * it_cd_loss + segl_weight * it_seg_loss
                epoch_loss_eval += it_loss.to("cpu").numpy()

        epoch_loss_eval /= len(dataset_val)
        
        if (epc != 0) and (min(validation_loss_list) > epoch_loss_eval):
            if segl_weight == 0.5:
                pass
            else:
                segl_weight   -= 0.05
                cdl_weight    += 0.05 
                print('balancing weight for loss has been modified... ')
                print('segment loss weight  : {} -> {}'.format(segl_weight+0.05, segl_weight))
                print('cd loss weight       : {} -> {}'.format(cdl_weight-0.05, cdl_weight))

        validation_loss_list.append(epoch_loss_eval)

        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU_val class change/epoch", scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch", scores_dictionary["F1_1"], epc)
        print("IoU class change for epoch {} is {}".format(epc, scores_dictionary["iou_1"]))
        print("F1 class change for epoch {} is {}".format(epc, scores_dictionary["F1_1"]))

    for epc in range(epochs):
        training_phase(epc)
        validation_phase(epc)
        early_stopping(validation_loss_list[-1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # scheduler step
        scheduler.step()

    plt.plot(list(map(int, epcs)), training_loss_list, color="red", label='Training loss')
    plt.plot(list(map(int, epcs)), validation_loss_list, color="blue", label='Evaluation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
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
    tr_dataset = loader_dict['spacenet7'](args.datapath, "train", False, cfg['data_config'])
    va_dataset = loader_dict['spacenet7'](args.datapath, "val", False, cfg['data_config'])
    data_loader_tr = DataLoader(tr_dataset, batch_size=cfg["params"]["batch_size"], shuffle=True)
    data_loader_va = DataLoader(va_dataset, batch_size=1, shuffle=False)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model()
    model = nn.DataParallel(model)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['params']['lr'], weight_decay=cfg['params']['weight_decay'], amsgrad=False)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # copy the configurations
    _ = shutil.copytree("./models",os.path.join(args.logpath, "models"),)

    train(
        data_loader_tr,
        data_loader_va,
        model,
        criterion,
        optimizer,
        scheduler,
        args.logpath,
        writer,
        epochs=cfg["params"]["epochs"],
        save_after=1,
        device=device,
        patience_limit=cfg["params"]["patience_limit"]
    )
    writer.close()

if __name__ == "__main__":
    run()
