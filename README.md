#Graphical Models for Identifying Pore-forming Proteins
This repository provides codes for paper *Nan Xu, Theodore W. Kahn, Theju Jacob, Yan Liu, Graphical Models for Identifying Pore-forming Protein*.

We put the extracted proteins in `./ExtractedProteins`. You can reproduce our experiments with the following codes, with the assumption that we are at location: `./project_package`.
## Requirements

This code package was tested with `python 3.7.3` on Linux. For other library dependencies, you can
1. First install [Anaconda](https://docs.anaconda.com/anaconda/install/) 
1. Refer to `environment.yaml` and install in terminal with Anaconda: 
    ```
    conda env create -f environment.yaml
    ```
1. Download the UniProt20 database from the following link (around 12G, takes about 30m to download):
    ```
    http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/uniprot20_2016_02.tgz
    ```
   1. uncompress zip file
   1. rename the folder to `uniprot20_2016_02`, ensure all files in this folder except `md5sum` have prefix `uniprot20_2016_02`
   1. move the folder to `packages/TGT_Package/databases`
  

## Run Experiment
1. Activate the environment:

    ```
    conda activate SCRFs
    ```
1. Change parameters in config.py
    
    The table below provides an overview of the parameters:

    | Name                  | Type            | Description                                                                                                                                                             | Default                                                  | Suggestion                                                                                                                                                                                                      |
    |-----------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | blastp_cmd_path       | string          | Path of Blastp package; Compute sequence identity; Details: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download                              | './packages/ncbi-blast-2.10.1+/bin/blastp'               | DO NOT Change                                                                                                                                                                                                   |
    | TGT_pkg_path          | string          | Path of TGT package; Run HHblits to generate A3M; Details: https://github.com/realbigws/TGT_Package                                                                     | './packages/TGT_Package'                                 | DO NOT Change; Ensure database UniProt20 downloaded                                                                                                                                                             |
    | Predict_Property_path | string          | Path of Predict Property package; Predict 3- and 8-digit secondary structures with TGT format protein profile; Details: https://github.com/realbigws/Predict_Property   | './packages/Predict_Property'                            | DO NOT Change                                                                                                                                                                                                   |
    | records_dir           | string          | Storage path of results                                                                                                                                                 | './records'                                              | Either relative (like the default) or absolute path should be fine.                                                                                                                                             |
    | sub_records_dir       | string          | Subfolder path under records_dir                                                                                                                                        | ''                                                       | DO NOT Change unless for debug use                                                                                                                                                                              |
    | task_type             | list of strings | Definition of tasks: selection for searching proteins with similar structure to group 1/2/3; filtering for combining results from different fasta files for group 1/2/3 | ['selection', 'filtering']                               | DO NOT Change                                                                                                                                                                                                   |
    | group_list            | list of char    | Definition of group index: we studied three groups of training PFT proteins with similar structures in this project                                                     | ['1', '2', '3']                                          | DO NOT Change                                                                                                                                                                                                   |
    | cur_task              | list of strings | Tasks to perform according to task_type definition                                                                                                                      | ['selection', 'filtering']                               | Normally use the default; Or ['selection']; Or ['filtering']                                                                                                                                                    |
    | selection_parameters  | dictionary      | Parameters for task selection                                                                                                                                           | -                                                        | -                                                                                                                                                                                                               |
    |                       |                 | group_index_list: list of group index to consider                                                                                                                       | ['1', '2', '3']                                          | Normally use the default; Or ['1'], ['2'], ['3'];  Or combination of any two group indices                                                                                                                      |
    |                       |                 | source_folder: source folder of fasta files                                                                                                                             | 'records/records_20201219'                               | Use your own folder path (either relative or absolute); NOTE: only proteins in files ending with '.fasta' will be considered                                                                                    |
    |                       |                 | target_folder: target folder of results                                                                                                                                 | 'selection'                                              | Normally use the default or specify your relative path; NOTE: absolute path is not supported here                                                                                                               |
    |                       |                 | seqlen_limit: list of integer, first is the allowed minimum sequence length, and second is the allowed maximum length                                                   | [50, 2000]                                               | Use your own sequence limitation values                                                                                                                                                                         |
    |                       |                 | pos_seq_identity_threshold: float,  testing proteins should be with sequence identity to training PFT proteins in each group smaller this value                         | 0.3                                                      | Use your own sequence identity threshold                                                                                                                                                                        |
    |                       |                 | PFTs_prob_threshold: float,  probability of being pore-forming toxins from ProtCNN model for testing proteins should be higher than this value                          | 0.5                                                      | Use your own probability threshold (higher than 0.5)                                                                                                                                                            |
    |                       |                 | mutual_seq_identity_threshold: float,  remove selected proteins with mutual sequence identity higher than this value to avoid repetition                                | 1                                                        | Use your own probability threshold (positive value not beyond 1)                                                                                                                                                |
    |                       |                 | batch_size: integer,  batch size for ProtCNN model                                                                                                                      | 100                                                      | Use your own batch size; NOTE: smaller than 1000 if gpu is used and the memory is around 12G                                                                                                                    |
    |                       |                 | num_threads: integer,  threads that run parallel to speed up computing                                                                                                  | 10                                                       | Use your own threads number; NOTE: observe your cpu and memory utilization, decrease the threads number if the server is overloaded                                                                             |
    |                       |                 | device: string,  device to run ProtCNN model                                                                                                                            | 'cpu'                                                    | Use your own device; NOTE: determine the device index if gpu is used, e.g., 'cuda:0' or 'cuda:1'                                                                                                                |
    | filtering_parameters  | dictionary      | parameters for task filtering                                                                                                                                           | -                                                        | -                                                                                                                                                                                                               |
    |                       |                 | source_folder: source folder of results (files ends with '_summary.csv') computed from selection task                                                                   | 'selection'                                              | Normally use the target_folder in selection_parameters                                                                                                                                                          |
    |                       |                 | target_folder: target folder of results                                                                                                                                 | 'filtering'                                              | Normally use the default or specify your relative path; NOTE: absolute path is not supported here;  intra-group results stored in 'group_x_summary.csv' and inter-group results stored in 'dataset_summary.csv' |
    |                       |                 | mutual_seq_identity_threshold: float,  remove selected proteins with mutual sequence identity higher than this value to avoid repetition                                | 1                                                        | Use your own probability threshold (positive value not beyond 1)                                                                                                                                                |
    | PosTest_threshold     | dictionary      | threshold for structure prediction scores:  keep the testing protein if its SCRFs score is higher than threshold                                                        | 2.72 for group_1, 424.75 for group_2, 145.38 for group_3 | DO NOT Change                                                                                                                                                                                                   |

1. run main.py and log stdout to log.txt
    ```
    python main.py >> logThe below commands assume we are at location: `./project_package`.
## Requirements

This code package was tested with `python 3.7.3` on Linux. For other library dependencies, you can
1. First install [Anaconda](https://docs.anaconda.com/anaconda/install/) 
1. Refer to `environment.yaml` and install in terminal with Anaconda: 
    ```
    conda env create -f environment.yaml
    ```
1. Download the UniProt20 database from the following link (around 12G, takes about 30m to download):
    ```
    http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/uniprot20_2016_02.tgz
    ```
   1. uncompress zip file
   1. rename the folder to `uniprot20_2016_02`, ensure all files in this folder except `md5sum` have prefix `uniprot20_2016_02`
   1. move the folder to `packages/TGT_Package/databases`
  

## Run Experiment
1. Activate the environment:

    ```
    conda activate SCRFs
    ```
1. Change parameters in config.py
    
    The table below provides an overview of the parameters:

    | Name                  | Type            | Description                                                                                                                                                           | Default                                                  | Suggestion                                                                                                                                                                                                      |
    |-----------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | blastp_cmd_path       | string          | Path of Blastp package; Compute sequence identity; Details: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download                            | './packages/ncbi-blast-2.10.1+/bin/blastp'               | DO NOT Change                                                                                                                                                                                                   |
    | TGT_pkg_path          | string          | Path of TGT package; Run HHblits to generate A3M; Details: https://github.com/realbigws/TGT_Package                                                                   | './packages/TGT_Package'                                 | DO NOT Change; Ensure database UniProt20 downloaded                                                                                                                                                             |
    | Predict_Property_path | string          | Path of Predict Property package; Predict 3- and 8-digit secondary structures with TGT format protein profile; Details: https://github.com/realbigws/Predict_Property | './packages/Predict_Property'                            | DO NOT Change                                                                                                                                                                                                   |
    | records_dir           | string          | Storage path of results                                                                                                                                               | './records'                                              | Either relative (like the default) or absolute path should be fine.                                                                                                                                             |
    | sub_records_dir       | string          | Subfolder path under records_dir                                                                                                                                      | ''                                                       | DO NOT Change unless for debug use                                                                                                                                                                              |
    | task_type             | list of strings | Definition of tasks: selection for searching proteins with similar structure to group1/2/3; filtering for combining results from different fasta files for group1/2/3 | ['selection', 'filtering']                               | DO NOT Change                                                                                                                                                                                                   |
    | group_list            | list of char    | Definition of group index: we have three groups of proteins with similar structures in this project                                                                   | ['1', '2', '3']                                          | DO NOT Change                                                                                                                                                                                                   |
    | cur_task              | list of strings | Tasks to perform according to task_type definition                                                                                                                    | ['selection', 'filtering']                               | Normally use the default; Or ['selection']; Or ['filtering']                                                                                                                                                    |
    | selection_parameters  | dictionary      | Parameters for task selection                                                                                                                                         | -                                                        | -                                                                                                                                                                                                               |
    |                       |                 | group_index_list: list of group index to consider                                                                                                                     | ['1', '2', '3']                                          | Normally use the default; Or ['1'], ['2'], ['3'];  Or combination of any two group indices                                                                                                                      |
    |                       |                 | source_folder: source folder of fasta files                                                                                                                           | 'records/records_20201219'                               | Use your own folder path (either relative or absolute); NOTE: only proteins in files ending with '.fasta' will be considered                                                                                    |
    |                       |                 | target_folder: target folder of results                                                                                                                               | 'selection'                                              | Normally use the default or specify your relative path; NOTE: absolute path is not supported here                                                                                                               |
    |                       |                 | seqlen_limit: list of integer, first is the allowed minimum sequence length, and second is the allowed maximum length                                                 | [50, 2000]                                               | Use your own sequence limitation values                                                                                                                                                                         |
    |                       |                 | pos_seq_identity_threshold: float,  testing proteins should be with sequence identity to positive proteins in each group smaller this value                           | 0.3                                                      | Use your own sequence identity threshold                                                                                                                                                                        |
    |                       |                 | PFTs_prob_threshold: float,  probability of being Pore-forming toxins from ProtCNN model for testing proteins should be higher than this value                        | 0.5                                                      | Use your own probability threshold (higher than 0.5)                                                                                                                                                            |
    |                       |                 | mutual_seq_identity_threshold: float,  remove selected proteins with mutual sequence identity higher than this value to avoid repetition                              | 1                                                        | Use your own probability threshold (positive value not beyond 1)                                                                                                                                                |
    |                       |                 | batch_size: integer,  batch size for ProtCNN model                                                                                                                    | 100                                                      | Use your own batch size; NOTE: smaller than 1000 if gpu is used and the memory is around 12G                                                                                                                    |
    |                       |                 | num_threads: integer,  threads that run parallel to speed up computing                                                                                                | 10                                                       | Use your own threads number; NOTE: observe your cpu and memory utilization, decrease the threads number if the server is overloaded                                                                             |
    |                       |                 | device: string,  device to run ProtCNN model                                                                                                                          | 'cpu'                                                    | Use your own device; NOTE: determine the device index if gpu is used, e.g., 'cuda:0' or 'cuda:1'                                                                                                                |
    | filtering_parameters  | dictionary      | parameters for task filtering                                                                                                                                         | -                                                        | -                                                                                                                                                                                                               |
    |                       |                 | source_folder: source folder of results (files ends with '_summary.csv') computed from selection task                                                                 | 'selection'                                              | Normally use the target_folder in selection_parameters                                                                                                                                                          |
    |                       |                 | target_folder: target folder of results                                                                                                                               | 'filtering'                                              | Normally use the default or specify your relative path; NOTE: absolute path is not supported here;  intra-group results stored in 'group_x_summary.csv' and inter-group results stored in 'dataset_summary.csv' |
    |                       |                 | mutual_seq_identity_threshold: float,  remove selected proteins with mutual sequence identity higher than this value to avoid repetition                              | 1                                                        | Use your own probability threshold (positive value not beyond 1)                                                                                                                                                |
    | PosTest_threshold     | dictionary      | threshold for structure prediction scores:  keep the testing protein if its SCRFs score is higher than threshold                                                      | 2.72 for group_1, 424.75 for group_2, 145.38 for group_3 | DO NOT Change                                                                                                                                                                                                   |

1. run main.py and log stdout to log.txt
    ```
    python main.py > log.txt 2>&1
    ```
   
## Toy Results
The fasta file `records/records_20201219/Crickmore_CryProteins.txt` is split into 10 files ending with `.fasta`. If we use the default parameters, i.e., the `config.py` is not modified, we are expecting to have results in `records/2021-01-xx-xx-xx-xx` (the folder with latest date)
1. group summaries: `records/2021-01-07-14-21-31/filtering/group_1_summary.csv`, proteins in each file sorted by `SCRFs Score`. No testing proteins are predicted to have similar structure as PFTs in `group 2` and `group 3`. Take the `group_1_summary.csv` as an example:

    |   | ID       | PFT Probability    | SCRFs Score | Sequence  |
    |---|----------|--------------------|-------------|-----------|
    | 0 | Cry21Ga1 | 0.9998382329940796 | 3.79        | MADLSN... |
    | 1 | Cry32Ga1 | 0.9999923706054688 | 2.73        | MYQNYN... |
1. dataset summaries: `records/2021-01-07-14-21-31/filtering/dataset_summary.csv` with proteins sorted by  `PFT Probability`. The summary file looks as follows:

    |   | ID       | PFT Probability    | Sequence  |
    |---|----------|--------------------|-----------|
    | 0 | Cry32Ga1 | 0.9999923706054688 | MYQNYN... |
    | 1 | Cry21Ga1 | 0.9998382329940796 | MADLSN... |
1. You can check `log.txt` to debug and remove it later together with target folder of task selection: `records/2021-01-07-14-21-31/selection`.txt 2>&1
    ```
   
## Toy Results
The fasta file `records/records_20201219/Crickmore_CryProteins.txt` is split into 10 files ending with `.fasta`. If we use the default parameters, i.e., the `config.py` is not modified, we are expecting to have results in `records/2021-01-xx-xx-xx-xx` (the folder with latest date)
1. group summaries: `records/2021-01-07-14-21-31/filtering/group_1_summary.csv`, proteins in each file sorted by `SCRFs Score`. No testing proteins are predicted to have similar structure as PFTs in `group 2` and `group 3`. Take the `group_1_summary.csv` as an example:

    |   | ID       | PFT Probability    | SCRFs Score | Sequence  |
    |---|----------|--------------------|-------------|-----------|
    | 0 | Cry21Ga1 | 0.9998382329940796 | 3.79        | MADLSN... |
    | 1 | Cry32Ga1 | 0.9999923706054688 | 2.73        | MYQNYN... |
1. dataset summaries: `records/2021-01-07-14-21-31/filtering/dataset_summary.csv` with proteins sorted by  `PFT Probability`. The summary file looks as follows:

    |   | ID       | PFT Probability    | Sequence  |
    |---|----------|--------------------|-----------|
    | 0 | Cry32Ga1 | 0.9999923706054688 | MYQNYN... |
    | 1 | Cry21Ga1 | 0.9998382329940796 | MADLSN... |
1. You can check `log.txt` to debug and remove it later together with target folder of task selection: `records/2021-01-07-14-21-31/selection`