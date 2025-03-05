#------------------------------------- Basic Imports --------------------------
import sys
import os
import time
import torch
import inspect
import configparser
from collections.abc import Sequence
#----------------------------------- Logging Imports --------------------------
#import logging
#----------------------------------- Logging config  --------------------------
"""
logging.basicConfig(filename='logs/werernns.log', encoding='utf-8', level=logging.DEBUG)
tellem = logging.getLogger(__name__)
"""

cfg = configparser.ConfigParser()
cfg.read('examples/settings.cfg')
print('[examples.utils] Shakespeare dataset location {0}'.format(cfg.get('TRAIN', 'the_data')))

@staticmethod
def splitFileThenWrite(path: str, split_percentage: int):
    """Splits a file based on the split_percentage into two files."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    n = len(data)
    the_result   = data[: int(n * (split_percentage/100))]
    val_leftover = data[int(n * (split_percentage/100)) :]

    name_a = path + "." + str(split_percentage) + "percent"
    name_b = path + "." + str(100-split_percentage) + "percent"

    try:
        with open(name_a, "x") as f:
            f.write(the_result)
        with open(name_b, "x") as f:
            f.write(val_leftover)
    except FileExistsError:
        print("I am not authorized to overwrite files...!")


def detach_tensors_in_list(the_tensor_lst):
    f_name = inspect.stack()[0][3]
    ndx = 0
    result = []
    for the_state in the_tensor_lst:
        #print( "State {0} is {1}".format(ndx, the_state) )
        result.append(the_state.detach().clone())
        ndx += 1
    return result


'''
This function takes two tensor size objects and the dimension on which they are 
to be concantenated. The size objects are iterated over and listed as match or 
not based on dimension and equality.
'''
def compare_tensor_sizes(tsA, tsB, dim):
    f_name = inspect.stack()[0][3]
    res = 1 
    resTxt = '' 
    ndx = 0
    if tsA == tsB:
        res = 1
        resTxt = 'Sizes are equal {0}'.format(tsA)
    elif len(tsA)==len(tsB):
        for a, b in zip(tsA,tsB):
            if a != b and dim==ndx:
                resTxt += '\n\t{0} ----- {1} dims not equal, but its not necessary '.format(a,b)
            elif a != b and dim != ndx:
                res = 0
                resTxt += '\n\t{0} ----- {1} dims not equal, this is no good'.format(a,b)
            else:
                resTxt += '\n\t{0} ----- {1}'.format(a, b)
            ndx += 1
    else:
        resTxt = 'Sizes are not equal\n\t{0}\n\t{1}'.format(tsA,tsB)
    return res, resTxt
            
def check_tensors(tensors, dim):
    f_name = inspect.stack()[0][3]
    result = 1
    resp = ''
    tsize = None
    for tensor in tensors:
        if tsize == None:
            tsize = tensor.shape
        elif torch.is_tensor(tensor):
            result, resp = compare_tensor_sizes(tsize, tensor.shape, dim)
        else:
            result, resp = check_tensors(tensor, dim)
    return result, resp

def torch_cat_with_check(tensors, dim=0):
    f_name = inspect.stack()[0][3]
    if not isinstance(tensors, Sequence):
        errTxt = '[{0}] I require a sequence'.format(f_name)
        #tellem.error(errTxt)
        raise TypeError(errTxt)
    else:
        if len(tensors) < 2:
            #tellem.error("[{0}] The list must have more than one tensor.".format(f_name))
            raise Exception("[{0}] The list must have more than one tensor.".format(f_name))
        else:
            tensors_match, errTxt = check_tensors(tensors, dim)
            if tensors_match == 0:
               # tellem.error(errTxt)
                raise Exception(errTxt)
    return torch.cat(tensors, dim)
