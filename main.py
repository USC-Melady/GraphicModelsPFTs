# Created by xunannancy at 2020/12/22
import config
from datetime import datetime
import os
if config.selection_parameters['device'] == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    device_index = config.selection_parameters['device'].split(':')[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_index
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import tensorflow as tf
import pickle
import subprocess
import shutil
import multiprocessing


from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIXML
from Bio import SeqIO
from io import StringIO

from tape import ProteinBertModel, TAPETokenizer

from pyseqlab.utilities import SequenceStruct, generate_trained_model


from utils import SeqSegAttributeExtractor, _construct_sequence_segment, compute_segmentation_score


def load_config():
    params_list = list()
    for task_name in config.cur_task:
        params = dict()
        params['PosTest_threshold'] = config.PosTest_threshold
        params['blastp_cmd_path'] = config.blastp_cmd_path
        params['TGT_pkg_path'] = config.TGT_pkg_path
        params['Predict_Property_path'] = config.Predict_Property_path

        params['records_dir'] = config.records_dir
        params['sub_records_dir'] = config.sub_records_dir
        params['cur_task'] = task_name

        if task_name == 'selection':
            params.update(config.selection_parameters)
        elif task_name == 'filtering':
            params.update(config.filtering_parameters)
        params_list.append(params)

    return params_list

def compute_seq_identity(params, subject_ID, query_ID, subject_seq, query_seq):
    subject_path = os.path.join(params['full_target_path'], '%s_%s_subject.fasta'%(subject_ID, query_ID))
    subject_seq = SeqRecord(Seq(subject_seq), id="subject")
    SeqIO.write(subject_seq, subject_path, "fasta")

    query_path = os.path.join(params['full_target_path'], '%s_%s_query.fasta'%(subject_ID, query_ID))
    query_seq = SeqRecord(Seq(query_seq), id="query")
    SeqIO.write(query_seq, query_path, "fasta")

    output = \
        NcbiblastpCommandline(cmd=params['blastp_cmd_path'], query=query_path,
                              subject=subject_path, outfmt=5)()[0]
    blast_result_record = NCBIXML.read(StringIO(output))
    if len(blast_result_record.alignments) == 0:
        os.remove(subject_path)
        os.remove(query_path)
        return -1e7

    else:
        for alignment in blast_result_record.alignments:
            for hsp in alignment.hsps:
                os.remove(subject_path)
                os.remove(query_path)
                return hsp.identities * 1.0 / hsp.align_length

    return

def _read_testing_proteins(info_dict):
    params = info_dict['params']
    source_folder = info_dict['source_folder']
    one_source_file_path = info_dict['one_source_file_path']

    protein_ID, seq_list = list(), list()
    with open(os.path.join(source_folder, one_source_file_path), 'r') as f:
        protein_data = f.readlines()
    for one_line in protein_data:
        # NOTE: make sure that the protein ID line starts with >
        if one_line.startswith('>'):
            protein_ID.append(one_line[1:].split()[0])
            seq_list.append('')
        else:
            if len(seq_list) > 0:
                seq_list[-1] += one_line.split()[0]
    assert len(protein_ID) == len(seq_list)
    # select sequences within length limit
    selected_index = [i for i in range(len(seq_list)) if
                      len(seq_list[i]) >= params['seqlen_limit'][0] and len(seq_list[i]) <=
                      params['seqlen_limit'][1]]

    with open(os.path.join(params['full_target_path'], one_source_file_path.split('.')[0] + '_ID_seq.json'),
              'w') as f:
        json.dump({'ID': np.array(protein_ID)[selected_index].tolist(),
                   'seq': np.array(seq_list)[selected_index].tolist()}, f, indent=4)

    return

def read_testing_proteins(params):
    # read testing proteins, sequentially process all the files in the source folder
    source_folder = params['full_source_path']
    print(f'Read protein sequences from {source_folder}')
    source_file_path_list = [i for i in os.listdir(source_folder) if i.endswith('.fasta')]
    param_dict_list = list()
    for i in range(len(source_file_path_list)):
        param_dict_list.append({
            'params': params,
            'source_folder': source_folder,
            'one_source_file_path': source_file_path_list[i],
        })

    iterations = int(np.ceil(len(source_file_path_list) / params['num_threads']))
    for iteration_index in tqdm(range(iterations)):
        pool = multiprocessing.Pool()
        pool.map(_read_testing_proteins, param_dict_list[iteration_index*params['num_threads']: (iteration_index+1)*params['num_threads']])
        pool.close()
        pool.join()
    return

def _compute_pos_seq_identity(info_dict):
    params = info_dict['params']
    one_ID_seq_file = info_dict['one_ID_seq_file']

    # read positive sequences
    if params['group_index'] == '1':
        positive_ID_seq_ss3_ss8 = json.load(open('public_materials/group_1/Thiol_cytolysin/ID_seq_ss3_ss8.json', 'r'))
    elif params['group_index'] == '2':
        positive_ID_seq_ss3_ss8 = json.load(open('public_materials/group_2/BB_PF/ID_seq_ss3_ss8.json', 'r'))
    elif params['group_index'] == '3':
        positive_ID_seq_ss3_ss8 = json.load(open('public_materials/group_3/Thiol_cytolysin/ID_seq_ss3_ss8.json', 'r'))


    cur_ID_seq_dict = json.load(
        open(os.path.join('/'.join(params['cur_full_target_path'].split('/')[:-1]), one_ID_seq_file), 'r'))
    cur_ID, cur_seq = cur_ID_seq_dict['ID'], cur_ID_seq_dict['seq']
    seq_identity_mat = list()
    for one_query_ID, one_query_seq in zip(cur_ID, cur_seq):
        seq_identity_mat.append(list())
        for one_subject_ID, one_subject_seq in zip(positive_ID_seq_ss3_ss8['ID'], positive_ID_seq_ss3_ss8['seq']):
            seq_identity_mat[-1].append(compute_seq_identity(params, one_subject_ID, one_query_ID, one_subject_seq, one_query_seq))
    seq_identity_mat = np.array(seq_identity_mat)

    # save sequence identity
    seq_identity_frame = pd.DataFrame(
        data=seq_identity_mat,
        index=cur_ID,
        columns=positive_ID_seq_ss3_ss8['ID']
    )
    seq_identity_frame.to_csv(
        os.path.join(params['cur_full_target_path'], one_ID_seq_file.split('_ID_seq.json')[0] + '_PosSeqIden.csv'))

    return

def compute_pos_seq_identity(params):
    print(f'Blastp sequence identity with positive proteins...')

    # sequence identity computation, with group positive proteins
    ID_seq_file_list = [i for i in os.listdir('/'.join(params['cur_full_target_path'].split('/')[:-1])) if i.endswith('_ID_seq.json')]

    param_dict_list = list()
    for i in range(len(ID_seq_file_list)):
        param_dict_list.append({
            'params': params,
            'one_ID_seq_file': ID_seq_file_list[i],
        })

    iterations = int(np.ceil(len(ID_seq_file_list) / params['num_threads']))
    for iteration_index in tqdm(range(iterations)):
        pool = multiprocessing.Pool()
        pool.map(_compute_pos_seq_identity, param_dict_list[iteration_index*params['num_threads']: (iteration_index+1)*params['num_threads']])
        pool.close()
        pool.join()

    return

def residual_block(x, filters, dil):
    shortcut = x
    bn1 = tf.keras.layers.BatchNormalization()(x)
    a1 = tf.keras.layers.Activation("relu")(bn1)
    conv1 = tf.keras.layers.Conv1D(filters, 3, dilation_rate=dil, padding="same")(
        a1)  # 1100 filters and 9 kernel size in ProtCNN

    bn2 = tf.keras.layers.BatchNormalization()(conv1)
    a2 = tf.keras.layers.Activation("relu")(bn2)
    conv2 = tf.keras.layers.Conv1D(filters, 1, padding="same")(a2)

    x = tf.keras.layers.Add()([conv2, shortcut])

    return x

def create_model(numclass=16, max_seqlen=600):
    input_x = tf.keras.layers.Input(shape=(max_seqlen+2, 768))
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(input_x)
    x = residual_block(x, 128, 1)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = residual_block(x, 128, 2)  # 4 blocks of these in ProtCNN
    x = tf.keras.layers.MaxPooling1D(max_seqlen+2)(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(numclass, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_x, outputs=out)

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # print(model.summary())
    return model

def compute_PFTs_prob(params):
    print(f'Pore-forming toxins probability computing...')
    # preserve sequences with identity smaller than ?? to perform further processing

    PDTs_prob_dict = pickle.load(open('public_materials/PFTs_prob.pkl', 'rb'))

    if torch.cuda.is_available() and 'cuda' in params['device']:
        device = torch.device(params['device'])
    else:
        device = torch.device('cpu')

    max_seqlen = 600
    default_seq = 'A' * max_seqlen
    batch_size = params['batch_size']

    ID_seq_file_list = [i for i in os.listdir('/'.join(params['cur_full_target_path'].split('/')[:-1])) if i.endswith('_ID_seq.json')]
    for file_index, one_ID_seq_file in enumerate(ID_seq_file_list):
        print(f'{file_index}/{len(ID_seq_file_list)} processing...')

        cur_ID_seq_dict = json.load(open(os.path.join('/'.join(params['cur_full_target_path'].split('/')[:-1]), one_ID_seq_file), 'r'))
        cur_ID, cur_seq = np.array(cur_ID_seq_dict['ID']), np.array(cur_ID_seq_dict['seq'])

        seq_identity_frame = pd.read_csv(os.path.join(params['cur_full_target_path'], one_ID_seq_file.split('_ID_seq.json')[0]+'_PosSeqIden.csv'))
        seq_identity_mat = seq_identity_frame.to_numpy()[:, 1:]
        selected_index = np.argwhere(np.max(seq_identity_mat, axis=-1) < params['pos_seq_identity_threshold']).reshape([-1])
        # if there is no testing proteins satisfying this condition
        if len(selected_index) == 0:
            print('No proteins in %s have similarity identity < %f'%(one_ID_seq_file.split('_ID_seq.json')[0], params['pos_seq_identity_threshold']))
            continue

        pos_selected_ID, pos_selected_seq = cur_ID[selected_index], cur_seq[selected_index]

        missing_ID, missing_seq = list(), list()
        for one_selected_ID, one_selected_seq in zip(pos_selected_ID, pos_selected_seq):
            if one_selected_ID not in PDTs_prob_dict:
                missing_ID.append(one_selected_ID)
                missing_seq.append(one_selected_seq)

        batch_num = int(np.ceil(len(missing_ID) / batch_size))
        overall_logit = list()
        model = ProteinBertModel.from_pretrained('bert-base').to(device)
        tokenizer = TAPETokenizer(vocab='iupac')

        for batch_index in tqdm(range(batch_num)):
            seq = missing_seq[batch_index * batch_size:(batch_index + 1) * batch_size]
            TAPEncoded_valid = list()
            # with torch.no_grad():

            # A BERT sequence has the following format: [CLS] X [SEP]
            validTAPE = [torch.from_numpy(np.array(tokenizer.encode(w[:max_seqlen].upper()))) for w in
                         [default_seq] + seq]
            TAPEvalid = torch.transpose(torch.nn.utils.rnn.pad_sequence(validTAPE, batch_first=False), 0, 1)
            for one_seq in TAPEvalid:
                TAPEncoded_valid.append(
                    model(one_seq.reshape([1, -1]).to(device))[0][0].cpu().detach().numpy())
            TAPEncoded_valid = np.array(TAPEncoded_valid)[1:]
            # print(f'emb: {TAPEncoded_valid.shape}')

            ProtCNN_model = create_model(numclass=2, max_seqlen=max_seqlen)
            ProtCNN_model.load_weights('public_materials/ProtCNN_model_binary.h5')

            test_predicted_logit = ProtCNN_model.predict(TAPEncoded_valid)[:, 1]
            overall_logit += test_predicted_logit.tolist()
        for one_missing_ID, one_missing_logit in zip(missing_ID, overall_logit):
            PDTs_prob_dict[one_missing_ID] = one_missing_logit


        with open(os.path.join(params['cur_full_target_path'], one_ID_seq_file.split('_ID_seq.json')[0]+'_ID_seq_PFTs_prob.json'), 'w') as f:
            json.dump({
                'ID': pos_selected_ID.tolist(),
                'seq': pos_selected_seq.tolist(),
                'prob': [PDTs_prob_dict[i] for i in pos_selected_ID],
            }, f, indent=4)

    with open('public_materials/PFTs_prob.pkl', 'wb') as f:
        pickle.dump(PDTs_prob_dict, f)

    return

def _conduct_ss_predict(info_dict):
    params = info_dict['params']
    one_ID_seq_file = info_dict['one_ID_seq_file']

    if not os.path.exists(os.path.join(params['TGT_pkg_path'], 'seqs_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0])):
        os.makedirs(os.path.join(params['TGT_pkg_path'], 'seqs_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))
    if not os.path.exists(os.path.join(params['TGT_pkg_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0])):
        os.makedirs(os.path.join(params['TGT_pkg_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))
    if not os.path.exists(os.path.join(params['Predict_Property_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0])):
        os.makedirs(os.path.join(params['Predict_Property_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))


    # secondary structure prediction
    ss3_ID_pred = pickle.load(open('public_materials/ss3_ID_pred.pkl', 'rb'))
    ss8_ID_pred = pickle.load(open('public_materials/ss8_ID_pred.pkl', 'rb'))


    cur_ID_seq_dict = json.load(open(os.path.join(params['cur_full_target_path'], one_ID_seq_file), 'r'))
    cur_ID, cur_seq, cur_PFTs_prob = np.array(cur_ID_seq_dict['ID']), np.array(cur_ID_seq_dict['seq']), np.array(
        cur_ID_seq_dict['prob'])

    selected_index = np.argwhere(cur_PFTs_prob > params['PFTs_prob_threshold']).reshape([-1])
    if len(selected_index) == 0:
        print('No proteins in %s have PFTs probability > %f' % (
        one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], params['PFTs_prob_threshold']))
        return
    PFTs_selected_ID, PFTs_selected_seq, PFTs_selected_prob = cur_ID[selected_index], cur_seq[selected_index], \
                                                              cur_PFTs_prob[selected_index]

    cur_ss3_pred, cur_ss8_pred = list(), list()

    for one_ID, one_seq in tqdm(zip(PFTs_selected_ID, PFTs_selected_seq), total=len(PFTs_selected_ID)):
        if one_ID not in ss3_ID_pred:
            with open(os.path.join(params['TGT_pkg_path'], 'seqs_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], f'{one_ID}.fasta'),
                      'w') as f:
                f.writelines('>%s\n' % one_ID)
                f.writelines(one_seq)

            with open('%s/seqs_%s/%s_tmp.txt' % (params['TGT_pkg_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID), 'w') as output:
                p = subprocess.Popen([
                    '%s/A3M_TGT_Gen.sh' % params['TGT_pkg_path'],
                    '-c', '4',
                    '-i', '%s/seqs_%s/%s.fasta' % (
                        params['TGT_pkg_path'],
                        one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0],
                        one_ID,
                    ),
                    '-o', '%s/output_%s/%s' % (
                        params['TGT_pkg_path'],
                        one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0],
                        one_ID
                    ),
                    '-d', 'uniprot20_2016_02'], stdout=output)
                p.communicate()
                p.wait()
                os.remove('%s/seqs_%s/%s_tmp.txt' % (params['TGT_pkg_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID))

            with open('%s/output_%s/%s_tmp.txt' % (params['Predict_Property_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID), 'w') as output:
                p = subprocess.Popen([
                    '%s/Predict_Property.sh' % params['Predict_Property_path'],
                    '-i', '%s/output_%s/%s/%s.tgt' % (
                        params['TGT_pkg_path'],
                        one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0],
                        one_ID,
                        one_ID
                    ),
                    '-o', '%s/output_%s/%s' % (
                        params['Predict_Property_path'],
                        one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0],
                        one_ID
                    )
                ], stdout=output)
                p.communicate()
                p.wait()
                os.remove('%s/output_%s/%s_tmp.txt' % (params['Predict_Property_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID))
            # read results
            with open(os.path.join('%s/output_%s/%s' % (params['Predict_Property_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID), '%s.ss3_simp' % one_ID),
                      'r') as f:
                data = f.readlines()
            assert data[0][1:-1] == one_ID
            ss3_ID_pred[one_ID] = data[-1].split()[0]

            with open(os.path.join('%s/output_%s/%s' % (params['Predict_Property_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID), '%s.ss8_simp' % one_ID),
                      'r') as f:
                data = f.readlines()
            assert data[0][1:-1] == one_ID
            ss8_ID_pred[one_ID] = data[-1].split()[0]

            # remove folders
            os.remove('%s/seqs_%s/%s.fasta' % (params['TGT_pkg_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID))
            shutil.rmtree('%s/output_%s/%s' % (params['TGT_pkg_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID))
            shutil.rmtree('%s/output_%s/%s' % (params['Predict_Property_path'], one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0], one_ID))
        cur_ss3_pred.append(ss3_ID_pred[one_ID])
        cur_ss8_pred.append(ss8_ID_pred[one_ID])

    with open(os.path.join(params['cur_full_target_path'],
                           one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0] + '_ID_seq_ss_pred.json'), 'w') as f:
        json.dump({
            'ID': PFTs_selected_ID.tolist(),
            'seq': PFTs_selected_seq.tolist(),
            'prob': PFTs_selected_prob.tolist(),
            'ss3': cur_ss3_pred,
            'ss8': cur_ss8_pred
        }, f, indent=4)


    with open('public_materials/ss3_ID_pred.pkl', 'wb') as f:
        pickle.dump(ss3_ID_pred, f)
    with open('public_materials/ss8_ID_pred.pkl', 'wb') as f:
        pickle.dump(ss8_ID_pred, f)

    shutil.rmtree(os.path.join(params['TGT_pkg_path'], 'seqs_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))
    shutil.rmtree(os.path.join(params['TGT_pkg_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))
    shutil.rmtree(os.path.join(params['Predict_Property_path'], 'output_%s'%one_ID_seq_file.split('_ID_seq_PFTs_prob.json')[0]))

    return

def conduct_ss_predict(params):
    print('Secondary Structure Prediction Computing...')

    ID_seq_file_list = [i for i in os.listdir(params['cur_full_target_path']) if i.endswith('_ID_seq_PFTs_prob.json')]

    param_dict_list = list()
    for i in range(len(ID_seq_file_list)):
        param_dict_list.append({
            'params': params,
            'one_ID_seq_file': ID_seq_file_list[i],
        })
    iterations = int(np.ceil(len(ID_seq_file_list) / params['num_threads']))
    for iteration_index in tqdm(range(iterations)):
        pool = multiprocessing.Pool()
        pool.map(_conduct_ss_predict, param_dict_list[iteration_index*params['num_threads']: (iteration_index+1)*params['num_threads']])
        pool.close()
        pool.join()

    return


def _semi_CRFs_workflow(info_dict):
    params = info_dict['params']
    one_ID_seq_file = info_dict['one_ID_seq_file']

    generic_attr_extractor = SeqSegAttributeExtractor()
    crf_percep = generate_trained_model('public_materials/group_%s/model_parts'%params['group_index'], generic_attr_extractor)
    decoding_method = 'viterbi'
    sep = "\t"

    output_dir = os.path.join(params['cur_full_target_path'], 'SCRFs_output_%s'%one_ID_seq_file.split('_ID_seq_ss_pred.json')[0])

    cur_ID_seq_prob_ss_pred_dict = json.load(open(os.path.join(params['cur_full_target_path'], one_ID_seq_file), 'r'))
    ID_list, seq_list, prob_list, ss3_list, ss8_list = np.array(cur_ID_seq_prob_ss_pred_dict['ID']), np.array(
        cur_ID_seq_prob_ss_pred_dict['seq']), np.array(cur_ID_seq_prob_ss_pred_dict['prob']), np.array(
        cur_ID_seq_prob_ss_pred_dict['ss3']), np.array(cur_ID_seq_prob_ss_pred_dict['ss8'])

    testing_seq_segment = _construct_sequence_segment(
        seq_info=seq_list.tolist(),
        label_info=[[0] * len(i) for i in seq_list],
        ss3_pred=ss3_list.tolist(),
        ss8_pred=ss8_list.tolist()
    )

    decoded_testing_sequences = dict()
    for testing_idx, one_testing_seq in enumerate(testing_seq_segment):
        cur_decoded = crf_percep.decode_seqs(decoding_method, output_dir, seqs=[one_testing_seq], sep=sep)
        decoded_testing_sequences[1 + testing_idx] = cur_decoded[1]
        shutil.rmtree(output_dir)
    testing_predictions = [''.join(decoded_testing_sequences[i]['Y_pred']) for i in
                           decoded_testing_sequences.keys()]
    SCRF_scores = compute_segmentation_score(params['group_index'], ID_list, seq_list, testing_predictions)

    with open(os.path.join(params['cur_full_target_path'],
                           one_ID_seq_file.split('_ID_seq_ss_pred.json')[0] + '_ID_seq_prob_SCRFScore.json'), 'w') as f:
        json.dump({
            'ID': ID_list.tolist(),
            'seq': seq_list.tolist(),
            'prob': prob_list.tolist(),
            'SCRF_scores': SCRF_scores.tolist(),
        }, f, indent=4)

    # save to csv, columns: ['ID', 'PFT Probability', 'SCRFs Score', 'Sequence']
    selected_index = np.argwhere(SCRF_scores > params['PosTest_threshold']['group_%s' % params['group_index']]).reshape(
        [-1])
    if len(selected_index) == 0:
        print('No proteins in %s have SCRF scores higher than %f' % (one_ID_seq_file.split('_ID_seq_ss_pred.json')[0],
                                                                     params['PosTest_threshold'][
                                                                         'group_%s' % params['group_index']]))
        return
    selected_ID, selected_seq, selected_prob, selected_SCRF_scores = ID_list[selected_index], seq_list[selected_index], \
                                                                     prob_list[selected_index], SCRF_scores[
                                                                         selected_index]

    sorted_index = np.argsort(selected_prob)[::-1]
    sorted_ID, sorted_seq, sorted_prob, sorted_SCRF_scores = selected_ID[sorted_index], selected_seq[sorted_index], \
                                                             selected_prob[sorted_index], selected_SCRF_scores[
                                                                 sorted_index]

    # compute mutual identity
    mutual_seq_identity_mat = -1 * np.ones([len(sorted_ID), len(sorted_ID)])
    for first_index, first_seq in enumerate(sorted_seq):
        for second_index, second_seq in enumerate(sorted_seq):
            if second_index <= first_index:
                continue
            mutual_seq_identity_mat[first_index][second_index] = compute_seq_identity(params, sorted_ID[first_index], sorted_ID[second_index], first_seq, second_seq)

    selected_testing_index = set(range(len(sorted_ID)))
    for one_testing_index in range(len(sorted_ID)):
        if one_testing_index not in selected_testing_index:
            continue
        removed_index = set(
            np.argwhere(mutual_seq_identity_mat[one_testing_index] >= params['mutual_seq_identity_threshold']).reshape(
                [-1]).tolist())
        selected_testing_index = selected_testing_index - removed_index
    selected_testing_index = sorted(list(selected_testing_index))
    sorted_ID, sorted_prob, sorted_SCRF_scores, sorted_seq = sorted_ID[selected_testing_index], sorted_prob[
        selected_testing_index], sorted_SCRF_scores[selected_testing_index], sorted_seq[selected_testing_index]

    sorted_frame = pd.DataFrame(
        data=list(zip(sorted_ID, sorted_prob, sorted_SCRF_scores, sorted_seq)),
        index=np.arange(len(sorted_ID)), columns=['ID', 'PFT Probability', 'SCRFs Score', 'Sequence']
    )
    sorted_frame.to_csv(
        os.path.join(params['cur_full_target_path'], one_ID_seq_file.split('_ID_seq_ss_pred.json')[0] + '_summary.csv'))

    return

def semi_CRFs_workflow(params):
    """
    threshold:
    group2 -
    group3 - 145.38
    :param params:
    :return:
    """

    # perform segmentation
    print(f'SCRFs segmenting...')

    ID_seq_file_list = [i for i in os.listdir(params['cur_full_target_path']) if i.endswith('_ID_seq_ss_pred.json')]

    param_dict_list = list()
    for i in range(len(ID_seq_file_list)):
        param_dict_list.append({
            'params': params,
            'one_ID_seq_file': ID_seq_file_list[i],
        })

    iterations = int(np.ceil(len(ID_seq_file_list) / params['num_threads']))
    for iteration_index in tqdm(range(iterations)):
        pool = multiprocessing.Pool()
        pool.map(_semi_CRFs_workflow, param_dict_list[iteration_index*params['num_threads']: (iteration_index+1)*params['num_threads']])
        pool.close()
        pool.join()
    return


def protein_selection(params):
    print('Task Selection...')
    read_testing_proteins(params) # parallel enabled
    for group_index in params['group_index_list']:
        params['group_index'] = group_index
        print('group_%s running...'%params['group_index'])
        params['cur_full_target_path'] = os.path.join(params['full_target_path'], 'group_%s'%params['group_index'])
        if not os.path.exists(params['cur_full_target_path']):
            os.makedirs(params['cur_full_target_path'])
        compute_pos_seq_identity(params) # parallel enabled
        compute_PFTs_prob(params) # consider GPU limitation, parallel disable
        conduct_ss_predict(params) # parallel enabled
        semi_CRFs_workflow(params) # parallel enabled
        print('-'*40)
        print('-'*40)
    return


def compute_mutual_seq_identity(summary_list):
    data_list = list()
    for one_file in summary_list:
        data_list.append(pd.read_csv(one_file).to_numpy()[:, 1:])

    # columns: ['ID', 'PFT Probability', 'SCRFs Score', 'Sequence']

    for second_file_index in range(1, len(data_list)):
        second_info = data_list[second_file_index]
        preserved_second_info = second_info.copy()
        for first_file_index in range(second_file_index):
            first_info = data_list[first_file_index]
            # remove identical ids
            preserved_index = [i for i in range(len(preserved_second_info)) if preserved_second_info[i][0] not in first_info[:, 0]]
            preserved_second_info = preserved_second_info[preserved_index]
            preserved_second_index = list()
            for second_index in range(len(preserved_second_info)):
                second_ID, second_seq = preserved_second_info[second_index][0], preserved_second_info[second_index][-1]
                keep_flag = True
                for first_index in range(len(first_info)):
                    first_ID, first_seq = first_info[first_index][0], first_info[first_index][-1]
                    cur_identity = compute_seq_identity(params, first_ID, second_ID, first_seq, second_seq)
                    if cur_identity >= params['mutual_seq_identity_threshold']:
                        keep_flag = False
                        break
                if keep_flag:
                    preserved_second_index.append(second_index)
            preserved_second_info = preserved_second_info[preserved_second_index]
        data_list[second_file_index] = preserved_second_info.copy()
    saved_data_list = np.concatenate(data_list, axis=0)
    return saved_data_list[:, 0], saved_data_list[:, 1], saved_data_list[:, 2], saved_data_list[:, 3]


def protein_filtering(params):
    print('Task filtering...')
    # assume the source_folder has group_1, group_2, group_3
    # group_wise: combine groups for each group, dataset_wise: combine proteins for all three groups

    # group_wise
    group_summary_dict = dict()
    dataset_summary_list = list()
    for group_index in ['1', '2', '3']:
        if os.path.exists(os.path.join(params['full_source_path'], 'group_%s'%group_index)):
            summary_path_list = [os.path.join(params['full_source_path'], 'group_%s'%group_index, i) for i in os.listdir(os.path.join(params['full_source_path'], 'group_%s'%group_index)) if i.endswith('_summary.csv')]
            if len(summary_path_list) > 0:
                group_summary_dict['group_%s'%group_index] = summary_path_list
        print(f'%d summaries for group_%s'%(len(group_summary_dict['group_%s'%group_index]) if 'group_%s'%group_index in group_summary_dict else 0, group_index))
    for group_index, group_summary_list in group_summary_dict.items():
        # ['ID', 'PFT Probability', 'SCRFs Score', 'Sequence']
        ID_list, prob_list, SCRF_scores_list, seq_list = compute_mutual_seq_identity(group_summary_list)
        # store in the target folder
        sorted_index = np.argsort(SCRF_scores_list)[::-1]
        frame = pd.DataFrame(
            data=list(zip(ID_list[sorted_index], prob_list[sorted_index], SCRF_scores_list[sorted_index], seq_list[sorted_index])),
            index=np.arange(len(ID_list)), columns=['ID', 'PFT Probability', 'SCRFs Score', 'Sequence']
        )
        frame.to_csv(os.path.join(params['full_target_path'], f'{group_index}_summary.csv'))
        dataset_summary_list.append(os.path.join(params['full_target_path'], f'{group_index}_summary.csv'))

    # dataset_wise
    print(f'%d summaries in %s'%(len(dataset_summary_list), params['full_source_path']))
    if len(dataset_summary_list) > 0:
        ID_list, prob_list, _, seq_list = compute_mutual_seq_identity(dataset_summary_list)
        sorted_index = np.argsort(prob_list)[::-1]
        frame = pd.DataFrame(
            data=list(zip(ID_list[sorted_index], prob_list[sorted_index], seq_list[sorted_index])),
            index=np.arange(len(ID_list)), columns=['ID', 'PFT Probability', 'Sequence']
        )
        frame.to_csv(os.path.join(params['full_target_path'], 'dataset_summary.csv'))

    return


if __name__ == '__main__':
    params_list = load_config()
    if params_list[0]['sub_records_dir'] == '':
        cur_path = os.path.join(params_list[0]['records_dir'], datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(cur_path)
    else:
        cur_path = os.path.join(params_list[0]['records_dir'], params_list[0]['sub_records_dir'])
        if not os.path.exists(cur_path):
            raise Exception(f'Record path {cur_path} does not exists!')

    with open(os.path.join(cur_path, 'params.json'), 'w') as f:
        json.dump(params_list, f, indent=4)

    for params in params_list:
        if os.path.exists(params['source_folder']):
            params['full_source_path'] = params['source_folder']
        else:
            params['full_source_path'] = os.path.join(cur_path, params['source_folder'])

        params['full_target_path'] = os.path.join(cur_path, params['target_folder'])
        if not os.path.exists(params['full_target_path']):
            os.makedirs(params['full_target_path'])

        if params['cur_task'] == 'selection':
            protein_selection(params)
        elif params['cur_task'] == 'filtering':
            protein_filtering(params)
