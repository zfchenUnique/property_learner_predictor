import json
import os
import pdb
import matplotlib.pyplot as plt


def parse_log_info(log_path):
    ep_list, loss_train_list, acc_train_list = [], [], []
    loss_val_list, acc_val_list = [], []
    ep_list_mass = []
    ep_list_charge = []
    acc_train_mass_list, acc_train_charge_list = [], []
    acc_val_mass_list, acc_val_charge_list = [], []
    with open(log_path, 'r') as fh:
        for line in fh.readlines():
            print(line)
            eles = line.split(' ')
            if len(eles) < 5:
                continue
            if "acc_val:" in eles:
                ep_list.append(int(eles[1]))
                loss_train_list.append(float(eles[3]))
                acc_train_list.append(float(eles[5]))
                loss_val_list.append(float(eles[7]))
                acc_val_list.append(float(eles[9]))
            elif "acc_val_mass:" in eles:
                ep_list_mass.append(int(eles[1]))
                acc_train_mass_list.append(float(eles[5]))
                acc_val_mass_list.append(float(eles[9]))
            elif "acc_val_charge:" in eles:
                ep_list_charge.append(int(eles[1]))
                acc_train_charge_list.append(float(eles[5]))
                acc_val_charge_list.append(float(eles[9]))
    return ep_list, loss_train_list, acc_train_list, loss_val_list, acc_val_list, ep_list_mass, acc_train_mass_list, acc_val_mass_list, ep_list_charge, acc_train_charge_list, acc_val_charge_list


def draw_training_log():
    fn_path = 'logs/exp_v15_encoder/log.txt'
    fn_noise_path = 'logs/exp_v15_encode_noise_0001/log.txt'
    fn_data_aug_path = 'logs/exp_v15_encode_ref_1_4/log.txt'
    fn_list = [fn_path, fn_noise_path, fn_data_aug_path]
    label_list = ['more data', 'with noise', 'with ref aug']
    for ty in ['mass', 'charge']:
        for idx, log_path in enumerate(fn_list):
            output = parse_log_info(log_path)
            ep_list_mass, acc_train_mass_list, acc_val_mass_list = output[-6:-3]
            ep_list_charge, acc_train_charge_list, acc_val_charge_list = output[-3:]
            if ty=='mass':
                plt.plot(ep_list_mass, acc_train_mass_list, label=label_list[idx], marker="o" )
                plt.plot(ep_list_mass, acc_val_mass_list, label=label_list[idx], marker="*" )
                img_path = 'mass_acc.png'
            else:
                plt.plot(ep_list_charge, acc_train_charge_list, label=label_list[idx], marker="o" )
                plt.plot(ep_list_charge, acc_val_charge_list, label=label_list[idx], marker="*" )
                img_path = 'charge_acc.png'
            plt.legend()
        plt.savefig(img_path)
    pdb.set_trace()

if __name__ == '__main__':
    draw_training_log()
