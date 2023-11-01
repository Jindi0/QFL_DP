import os
import argparse
from data_helper import load_dataset
import numpy as np
from client import Client
from util import init_log
from qiskit import Aer, IBMQ
from qiskit.utils import QuantumInstance
import qiskit_aer.noise as noise

def args_parser(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='fl30_c2', help='task name')
    parser.add_argument('--client_n', type=int, default=6, help='the number of training samples')

    parser.add_argument('--local_epoch', type=int, default=30, help='the number of iterations during local training')
    parser.add_argument('--global_round', type=int, default=2000, help='the number of global training round')

    parser.add_argument('--noisy', type=bool, default=False, help='if on noisy machine')
    parser.add_argument('--q_num', type=int, default=4, help='the input size')
    
    args = parser.parse_args()
    return args

args = args_parser()

savepath = './QFL/save_noisegates/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
    os.mkdir(savepath + 'data/')
task_path = savepath + '{}/'.format(args.task)
if not os.path.exists(task_path):
    os.mkdir(task_path)
    os.mkdir(task_path + 'client/')
    os.mkdir(task_path + 'client/ckps/')
    os.mkdir(task_path + 'checkpoints/')
# The avaiable backends
# each client uses a backend
backends=['ibm_perth', 'ibm_nairobi', 'ibm_lagos']


   

if __name__ == "__main__":
    # client_n = 6 for reduce the size of dataset of each client
    client_data_train_x, client_data_train_y, client_data_test_x, client_data_test_y = load_dataset('./QFL/', args.client_n, args.q_num)
    args.client_n = 3  # we only use 3 clients in QFL
    f, sheet = init_log(args.client_n)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    
    # Create client objects
    clients = []
    for i in range(args.client_n):
        if args.noisy:
            device = provider.get_backend(backends[i])  # Load the noise model of the backclient
            properties = device.properties()
            noise_model = noise.NoiseModel.from_backend(properties)
            quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), noise_model=noise_model)
        else:
            # noiseless training
            quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))

        client = Client(client_id=i, 
                        train_x=client_data_train_x[i], 
                        train_y=client_data_train_y[i], 
                        test_x=client_data_test_x[i], 
                        test_y=client_data_test_y[i], 
                        init_point = np.load(savepath + 'init_point.npy'),
                        q_num=args.q_num,
                        local_ep=args.local_epoch,
                        save_path=task_path+'client/',
                        quantum_inst=quantum_instance)
        
        clients.append(client)

    # Federated learning
    for r in range(args.global_round):
        print('Global round: {}'.format(r))
        local_model = []
        for i in range(args.client_n):
            clients[i].train_locally()
            test_acc = clients[i].evaluation()
            sheet.write(r+1, i*2+2, test_acc)
            f.save(task_path + 'records.xls')
            
            local_model.append(clients[i].get_parameters())
        local_model = np.array(local_model)

        global_model = np.mean(local_model, axis=0)
        np.save(task_path + 'checkpoints/global_ckp_{}.npy'.format(r), global_model)

        for i in range(args.client_n):
            clients[i].set_parameters(global_model)

        # record local training loss
        if r % 10 == 0:
            for i in range(args.client_n):
                local_loss = clients[i].get_localloss()
                for j in range(len(local_loss)):
                    sheet.write(j+1, args.client_n*2 +i+3, local_loss[j])
                f.save(task_path + 'records.xls')
    







    



