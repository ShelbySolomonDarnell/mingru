#------------------------------------- Basic Imports --------------------------
import datetime
import os
import pathlib
import re
import string
import pytz
import sys
import time
from typing import Any, Dict, Tuple
from venv import logger
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


def detach_tensors_in_list(the_tensor_container):
    """Detach tensors from computation graph.
    
    This function handles tensors in lists or tuples, making it compatible with both
    MinGRU (which uses lists) and MinLSTM (which uses tuples of lists).
    
    The function preserves the exact structure of the input container, ensuring
    that data is returned in the same format it was received.
    
    Args:
        the_tensor_container: A list or tuple of tensors, or a single tensor
        
    Returns:
        Container of the same type with detached tensors
    """
    f_name = inspect.stack()[0][3]
    
    # Handle None case
    if the_tensor_container is None:
        return None
        
    # Handle single tensor case
    if torch.is_tensor(the_tensor_container):
        return the_tensor_container.detach().clone()
        
    # Handle tuple case (for MinLSTM which returns (h, c))
    if isinstance(the_tensor_container, tuple):
        # Preserve tuple structure exactly
        return tuple(detach_tensors_in_list(item) for item in the_tensor_container)
    
    # Handle list case (for both MinGRU and elements of MinLSTM's tuple)
    if isinstance(the_tensor_container, list):
        list_length = len(the_tensor_container)
        #print(f"[{f_name}] Processing list with length: {list_length}")
        
        # Preserve list structure exactly
        result = []
        for ndx, the_state in enumerate(the_tensor_container):
            if torch.is_tensor(the_state):
                result.append(the_state.detach().clone())
            else:
                # Handle nested containers (like lists within lists)
                result.append(detach_tensors_in_list(the_state))
        return result
        
    # If we get here, we have an unsupported type
    raise TypeError(f"Unsupported container type: {type(the_tensor_container)}")


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


######################################################################
################ Checkpoint Related Functions ########################
######################################################################

@staticmethod
def load_model_checkpoint(
    load_checkpoint_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """Loads the optimizer state dict and model state dict from the load_checkpoint_dir
    into the passed model and optimizer. Searches for the most recent checkpoint to
    load from

    Args:
        load_checkpoint_dir (pathlib.Path):
            The base checkpoint directory to load from
        model (torch.nn.Module):
            The model to load the checkpoint weights into
        optimizer (torch.optim.Optimizer):
            The optimizer to load the checkpoint weigths into

    Returns:
        Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
            The checkpoint step, model with state_dict loaded and
            optimizer with state_dict loaded

    """
    logger.info(
        f"Loading model and optimizer checkpoint from {load_checkpoint_dir}")
    checkpoint_files = list(
        filter(
            lambda path: re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name) is
            not None,
            load_checkpoint_dir.glob("*.pt"),
        ))
    assert len(checkpoint_files) > 0, "No checkpoints found in directory"
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda path: int(
            re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name).group("iter_no")
        ),
    )
    latest_checkpoint_path = checkpoint_files[-1]
    checkpoint_step = int(
        re.search(r"iter_(?P<iter_no>\d+)\.pt",
                  latest_checkpoint_path.name).group("iter_no"))

    state_dict = torch.load(latest_checkpoint_path)
    model.load_state_dict(state_dict["model"], strict=True)
    optimizer.load_state_dict(state_dict["optimizer"])
    logger.info(
        f"Loading model and optimizer checkpoints done. Loaded from {latest_checkpoint_path}"
    )
    return checkpoint_step, model, optimizer


######################################################################
########### Experiment Management Related Functions ##################
######################################################################


def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix]
                   for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]) -> pathlib.Path:
    """Create an experiment directory and save all arguments in it.
    Additionally, also store the githash and gitdiff. Finally create
    a directory for `Tensorboard` logs. The structure would look something
    like
        checkpoint_dir
            `-experiment-name
                |- hparams.json
                |- githash.log
                |- gitdiff.log
                `- tb_dir/

    Args:
        checkpoint_dir (pathlib.Path):
            The base checkpoint directory
        all_arguments (Dict[str, Any]):
            The arguments to save

    Returns:
        pathlib.Path: The experiment directory
    """
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Central"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog)
    except sh.ErrorReturnCode_128:
        logger.info("Seems like the code is not running from"
                    " within a git repo, so hash will"
                    " not be stored. However, it"
                    " is strongly advised to use"
                    " version control.")
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff)
    except sh.ErrorReturnCode_129:
        logger.info("Seems like the code is not running from"
                    " within a git repo, so diff will"
                    " not be stored. However, it"
                    " is strongly advised to use"
                    " version control.")
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir()
    return exp_dir