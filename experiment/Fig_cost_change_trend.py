import numpy as np

from   search.utils.utils    import BFS, decode, iter_layer_compute, load_param
import yaml

gene_path    = '/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt'
# Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_cpu.yaml'
# Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_gpu.yaml'
# Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_flops.yaml'
# Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_params.yaml'
# Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_edge-gpu.yaml'
Param_path   = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/20_06_25_Res_Group_Spa_MBConv_Latency_edge-cpu.yaml'

print(Param_path)

iterations   =  5
add_decoder  = 'True'
add_encoder0 = 'True'
gene         = decode(gene_path, iterations)
skip_codes   = BFS(iterations, gene, add_decoder, add_encoder0)  # check if a block is skipped or not

with open(Param_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

conv_name  = []
op_value   = []
for key, item in config.items():
    if 'ResCon'      in key:
        conv_name.append(key[0:-5] + '_' + 'kernel:' + item.split('kernel:')[1].split('-')[0])
    elif 'GroupConv' in key:
        conv_name.append(key[0:-5] + '_' + 'group:' + item.split('group:')[1].split(',')[0])
    elif 'SpaCon'    in key:
        conv_name.append(key[0:-5] + '_' + 'kernel:' + item.split('kernel:')[1].split('-')[0])
        if 'SpaCon4' in key[0:-5] + '_' + 'kernel:' + item.split('kernel:')[1].split('-')[0]:
            print()
    elif 'MBConv'    in key:
        conv_name.append(key[0:-5] + '_' + 'expand:' + item.split('expand:')[1].split('-')[0])

    op_value.append(float(item.split('value:')[1]))

total_op_value = np.zeros(12)
conv_name_final = [True for i in range(12)]
for num in range(len(conv_name)):
    conv_name_final[num%12]  = conv_name[num]
    total_op_value[num%12]  += op_value[num]

for num in range(len(conv_name_final)):
    print(conv_name_final[num])
for num in range(len(conv_name_final)):
    print(total_op_value[num])

avg_op_value = np.zeros(4)
for num in range(0,len(total_op_value)):
    avg_op_value[int(num/3)]+= total_op_value[num]

print(avg_op_value/3)

