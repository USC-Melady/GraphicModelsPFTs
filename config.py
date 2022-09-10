# Created by xunannancy at 2020/12/22
blastp_cmd_path = './packages/ncbi-blast-2.10.1+/bin/blastp'
TGT_pkg_path = './packages/TGT_Package'# remember to download uniprot20 dataset
Predict_Property_path = './packages/Predict_Property'

records_dir = './records'
sub_records_dir = ''

task_type = [
    'selection',
    'filtering',
]
group_list = ['1', '2', '3']

cur_task = task_type

selection_parameters = {
    'group_index_list': group_list,
    'source_folder': 'records/records_20201219', # consider all files ending with .fasta in this folder
    'target_folder': 'selection',
    'seqlen_limit': [50, 2000],
    'pos_seq_identity_threshold': 0.3,
    'PFTs_prob_threshold': 0.5,
    'mutual_seq_identity_threshold': 1,
    'batch_size': 100,
    'num_threads': 10,
    'device': 'cpu',# or 'cuda:0'
}


filtering_parameters = {
    'source_folder': 'selection',
    'target_folder': 'filtering',
    'mutual_seq_identity_threshold': 1,
}

# NOTE: do not change
PosTest_threshold = {
    'group_1': 2.72,
    'group_2': 424.75,
    'group_3': 145.38,
}
