import os
import time
import numpy as np



def select_gpu(tmp, memory_per = 50000):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str('3')
    while 1: 
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}_1'.format(tmp))
        memory_gpu = np.array([int(x.split()[2]) for x in open('{}_1'.format(tmp), 'r').readlines()])
        os.system('rm {}_1'.format(tmp))
        os.system('nvidia-smi -q -d Utilization | grep -A1 Utilization | grep % >{}_2'.format(tmp))
        util_gpu = np.array([int(x.split()[2]) for x in open('{}_2'.format(tmp), 'r').readlines()])
        os.system('rm {}_2'.format(tmp))
        available_gpu = np.where((memory_gpu > memory_per) * (util_gpu <= 98))[0]
        if np.random.rand() < len(available_gpu) / 4: 
            memory_available = memory_gpu[available_gpu] / np.log(util_gpu[available_gpu]+1.1)
            memory_available = memory_available / np.sum(memory_available)
            random = np.random.rand()
            for i in range(len(memory_available)): 
                if random <= memory_available[i]: 
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu[i])
                    return
                else: 
                    random -= memory_available[i]
        time.sleep(np.random.exponential(60/(len(available_gpu)+1)) + time.time() * 10**15 % 60)

def gpu_memory_keeper(tmp, memory_per = 50000): 
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}_1'.format(tmp))
    memory_gpu = np.array([int(x.split()[2]) for x in open('{}_1'.format(tmp), 'r').readlines()])
    os.system('rm {}_1'.format(tmp))
    available_gpu = np.where(memory_gpu > memory_per)[0]
    while 1: 
        if np.sum(memory_gpu[available_gpu]) > memory_per * 8: 
            return
        else: 
            time.sleep(time.time() * 10**15 % 60)

if __name__ == '__main__': 
    gpu_memory_keeper('asd')