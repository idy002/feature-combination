import os

from .Dataset import Dataset


class MovieLens(Dataset):
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
    num_fields = 12
    max_length = 12
    num_features = 1139
    feat_names = ['age', 'occupation', 'zipcode', 'release_day', 'release_month', 'release_year',
                    'genre', 'hour', 'day', 'weekday', 'month', 'year']

    feat_min = [0, 67, 88, 883, 914, 926, 1026, 1045, 1069, 1100, 1107, 1119]
    feat_max = [66, 87, 882, 913, 925, 1025, 1044, 1068, 1099, 1106, 1118, 1138]
    feat_sizes = [67, 21, 795, 31, 12, 100, 19, 24, 31, 7, 12, 20]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MovieLens')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True):
        """
        collect meta information, and produce hdf files if not exists
        :param initialized: write feature and hdf files if True
        """
        self.initialized = initialized
        if not self.initialized:
            print('Got raw MovieLens data, initializing...')
            self.feat_names, self.feat_min, self.feat_max, self.feat_sizes = \
                self.collect_feature_info(raw_file='feature.txt')
            print(self.feat_names)
            print(self.feat_min)
            print(self.feat_max)
            print(self.feat_sizes)
            if self.max_length is None or self.num_features is None:
                print('Getting the maximum length and # features...')
                min_train_length, max_train_length, max_train_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'train.txt'))
                min_test_length, max_test_length, max_test_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'test.txt'))
                self.max_length = max(max_train_length, max_test_length)
                self.num_features = max(max_train_feature, max_test_feature) + 1
            print('max length = %d, # features = %d' % (self.max_length, self.num_features))

            self.train_num_of_parts = self.raw_to_feature(raw_file='train.txt',
                                                          input_feat_file='train_input',
                                                          output_feat_file='train_output')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)
            self.test_num_of_parts = self.raw_to_feature(raw_file='test.txt',
                                                         input_feat_file='test_input',
                                                         output_feat_file='test_output')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir)

        print('Got hdf MovieLens data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        print('Initialization finished!')

    def collect_feature_info(self, raw_file):
        print('Collecting feature info from', raw_file)
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feat_names = []
        feat_min = []
        feat_max = []
        feat_sizes = []
        with open(raw_file, "r") as rin:
            features = rin.readline().strip().split(' ')
            cnt = 0
            for feature in features:
                name, size = tuple(feature.split(':'))
                size = int(size)
                feat_names.append(name)
                feat_sizes.append(size)
                feat_min.append(cnt)
                cnt += size
                feat_max.append(cnt - 1)
        return feat_names, feat_min, feat_max, feat_sizes

    def raw_to_feature(self, raw_file, input_feat_file, output_feat_file):
        """
        Transfer the raw data to feature data. using static method is for consistence 
            with multi-processing version, which can not be packed into a class
        :param raw_file: The name of the raw data file.
        :param input_feat_file: The name of the feature input data file.
        :param output_feat_file: The name of the feature output data file.
        :return:
        """
        print('Transferring raw', raw_file, 'data into feature', raw_file, 'data...')
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feature_input_file_name = os.path.join(self.feature_data_dir, input_feat_file)
        feature_output_file_name = os.path.join(self.feature_data_dir, output_feat_file)
        line_no = 0
        cur_part = 0
        if self.block_size is not None:
            fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
            fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')
        else:
            fin = open(feature_input_file_name, 'w')
            fout = open(feature_output_file_name, 'w')
        with open(raw_file, 'r') as rin:
            for line in rin:
                line_no += 1
                if self.block_size is not None and line_no % self.block_size == 0:
                    fin.close()
                    fout.close()
                    cur_part += 1
                    fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
                    fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')

                fields = line.strip().split()
                y_i = fields[0]
                X_i = list(map(lambda x: int(x.split(':')[0]), fields[1:]))
                fout.write(y_i + '\n')
                first = True

                if len(X_i) > self.max_length:
                    X_i = X_i[:self.max_length]
                elif len(X_i) < self.max_length:
                    X_i.extend([self.num_features + 1] * (self.max_length - len(X_i)))

                for item in X_i:
                    if first:
                        fin.write(str(item))
                        first = False
                    else:
                        fin.write(' ' + str(item))
                fin.write('\n')
        fin.close()
        fout.close()
        return cur_part + 1

    @staticmethod
    def get_length_and_feature_number(file_name):
        """
        Get the min_length max_length and max_feature of data.
        :param file_name: The file name of input data.
        :return: the tuple (min_length, max_length, max_feature)
        """
        max_length = 0
        min_length = 99999
        max_feature = 0
        line_no = 0
        with open(file_name) as fin:
            for line in fin:
                line_no += 1
                if line_no % 100000 == 0:
                    print('%d lines finished.' % (line_no))
                fields = line.strip().split()
                X_i = list(map(lambda x: int(x.split(':')[0]), fields[1:]))
                max_feature = max(max_feature, max(X_i))
                max_length = max(max_length, len(X_i))
                min_length = min(min_length, len(X_i))
        return min_length, max_length, max_feature
