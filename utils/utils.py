import numpy as np


def get_gpu_memory(nvidia_smi):
    
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    
    return info.used/1000000000


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
                           
def load_weights(weight_file, classes):
   # Load the weight matrix.
   rows, cols, values = load_table(weight_file)
   assert(rows == cols)
   num_rows = len(rows)

   # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
   num_classes = len(classes)
   weights = np.zeros((num_classes, num_classes), dtype=np.float64)
   for i, a in enumerate(rows):
       if a in classes:
           k = classes.index(a)
           for j, b in enumerate(rows):
               if b in classes:
                   l = classes.index(b)
                   weights[k, l] = values[i, j]

   return weights


# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False