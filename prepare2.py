import time
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from mosgim2.data.loader import LoaderTxt, LoaderHDF
from mosgim2.data.tec_prepare import(process_data,
                                     combine_data,
                                     calculate_seed_mag_coordinates_parallel,
                                     get_data,
                                     sites,
                                     DataSourceType)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data from txt')
    parser.add_argument('--data_path', 
                        type=Path, 
                        help='Path to txt data', default = Path('/home/teufel/PycharmProjects/mosgim2/tmp') )   #Path('/home/teufel/PycharmProjects/MosGIM_2layer/tec-suite/tec/2017/002a/')
    parser.add_argument('--data_source', 
                        type=DataSourceType, 
                        help='Type of input data', default = 'hdf')   #txt
    parser.add_argument('--res_file',  
                        type=Path,
                        default=Path('/home/teufel/PycharmProjects/mosgim2/tmp/input1_hdf.npz'),
                        help='Path to file with results, for magnetic lat')
    parser.add_argument('--ipph1',  
                        type=float,
                        default=300000.,
                        help='height of first shell [m]')
    parser.add_argument('--ipph2',  
                        type=float,
                        default=750000.,
                        help='height of first shell [m]')

    args = parser.parse_args()

    IPPh_layer1 = args.ipph1
    IPPh_layer2 = args.ipph2

    if args.data_source == DataSourceType.hdf:
        loader = LoaderHDF(hdf_path=args.data_path, IPPh1 = IPPh_layer1, IPPh2 = IPPh_layer2)
        data_generator = loader.generate_data(sites=sites)
    if args.data_source == DataSourceType.txt:
        loader = LoaderTxt(root_dir=args.data_path, IPPh1 = IPPh_layer1, IPPh2 = IPPh_layer2)
        data_generator = loader.generate_data(sites=sites)


    data = process_data(data_generator)
    data_chunks = combine_data(data, nchunks=1)
    print('Start magnetic calculations...')
    st = time.time()
    result = calculate_seed_mag_coordinates_parallel(data_chunks)

    data = get_data(result)
    data['IPPh_layer1'] = IPPh_layer1
    data['IPPh_layer2'] = IPPh_layer2

    print(f'Done, took {time.time() - st}')
    np.savez(args.res_file, **data)
