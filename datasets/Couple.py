import os
import json

from .Dataset import Dataset


class Couple(Dataset):
    block_size = 2000000
    train_num_of_parts = 1
    test_num_of_parts = 1
    train_size = 0
    test_size = 0
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    _fields = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    num_fields = len(_fields)
    max_length = num_fields
    num_features = sum(_fields)
    
    meta = None

    feat_names = ["f" + str(i) for i in range(num_fields)]
    feat_sizes = [v for v in _fields]
    feat_min = [0 for i in range(num_fields)]
    for i in range(1,num_fields) :
        feat_min[i] = feat_min[i-1] + feat_sizes[i-1]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/Couple')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')
    meta_file_path = os.path.join(data_dir, 'meta.txt')

    def __init__(self, initialized=True):
        self.initialized = initialized


        if not self.initialized:
            if Couple.meta == None:
                with open(self.meta_file_path, "r") as meta_file:
                    Couple.meta = json.load(meta_file)
            print('Got raw Couple data, initializing...')
            print('max length = %d, # feature = %d' % (self.max_length, self.num_features))
            self.train_num_of_parts = self.raw_to_feature(raw_file='raw.train.svm',
                                                          input_feat_file='train_input',
                                                          output_feat_file='train_output')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)
            self.test_num_of_parts = self.raw_to_feature(raw_file='raw.test.svm',
                                                         input_feat_file='test_input',
                                                         output_feat_file='test_output')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)

        print('Got hdf Couple data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        print('Initialization finished!')

    def raw_to_feature(self, raw_file, input_feat_file, output_feat_file):
        print('Transferring raw', raw_file, 'data into feature', raw_file, 'data...')
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feat_fin_name = os.path.join(self.feature_data_dir, input_feat_file)
        feat_fout_name = os.path.join(self.feature_data_dir, output_feat_file)
        line_no = 0
        cur_part = 0
        if self.block_size is not None:
            fin = open(feat_fin_name + '.part_' + str(cur_part), 'w')
            fout = open(feat_fout_name + '.part_' + str(cur_part), 'w')
        else:
            fin = open(feat_fin_name, 'w')
            fout = open(feat_fout_name, 'w')
        with open(raw_file, 'r') as rin:
            for line in rin:
                line_no += 1
                if self.block_size is not None and line_no % self.block_size == 0:
                    fin.close()
                    fout.close()
                    cur_part += 1
                    fin = open(feat_fin_name + '.part_' + str(cur_part), 'w')
                    fout = open(feat_fout_name + '.part_' + str(cur_part), 'w')

                fields = line.strip().split()
                y_i = fields[0]
                X_i = map(lambda x: x.split(':')[0], fields[1:])
                fout.write(y_i + '\n')
                fin.write(','.join(X_i))
                fin.write('\n')
        fin.close()
        fout.close()
        return cur_part + 1
