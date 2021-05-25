import h5py

def load_model(h5_path):
    '''
    Load h5 model
    Arg(s):
        h5_path : str
            path where .hdf5 model is stored (extension included)
    Returns:
        HDF5 group
    '''
    return h5py.File(h5_path, 'r')

def print_names(name):
    print(name)
    return None

def get_groups(h5py_file):
    '''
    Output 
    '''
    # group_dict = {}
    # keys = list(h5py_file.keys())
    # for key in keys:
    #     group_dict[key] = get_groups_recursive()
    h5py_file.visit(print_names)

if __name__ == "__main__":
    file_path = 'keras_models/keras_model.h5'
    f = load_model(file_path)
    get_groups(f)