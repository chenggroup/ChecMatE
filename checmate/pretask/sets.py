import json, yaml
from pathlib import Path
from os import path
from tabulate import tabulate


def load_config(fpath:str):
    """
    Load config file.
    """

    fmt = Path(fpath).suffix
    with open(fpath, "r", encoding="utf8") as f:
        if fmt == ".yaml":
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        elif fmt == ".json":
            config = json.load(f)
    
        else:
            raise TypeError("An unsupported file format. (Only json and yaml formats are supported)")

    return config


def update_dict(old_dict:dict, new_dict:dict):
    """
    recursively update a dict.
    
    Parameters
    ----------
    old_dict: 
        a old dictionary.
    new_dict: 
        a new dictionary contains what needs to be updated.

    Return
    ------
        None.
    """
    for k, v in new_dict.items():
        if (k in old_dict and isinstance(old_dict[k], dict) and isinstance(new_dict[k], dict)):
            update_dict(old_dict[k], new_dict[k])

        else:
            old_dict_key = {}
            for i in old_dict.keys():
                old_dict_key[i.upper()] = i

            if k.upper() in old_dict_key.keys():
                old_dict[old_dict_key[k.upper()]] = new_dict[k]

            else:
                old_dict[k] = new_dict[k]


def iterdict(input_dict:dict, out_list:list, loop_idx:int):
    """ 
    recursively generate a list of strings for further 
    print out CP2K input file

    Parameters
    ----------
    input_dict: 
        dictionary for CP2K input parameters
    out_list: 
        list of strings for printing
    loop_idx: 
        record of loop levels in recursion

    Return
    ------
        out_list
    """
    if len(out_list) == 0:
        out_list.append("\n")

    start_idx = len(out_list) - loop_idx - 2
    for k, v in input_dict.items():
        k = str(k)  # cast key into string

        #if value is dictionary
        if isinstance(v, dict):
            out_list.insert(-1 - loop_idx, "&" + k)
            out_list.insert(-1 - loop_idx, "&END " + k)
            iterdict(v, out_list, loop_idx + 1)

        # if value is list
        elif isinstance(v, list):
            for _v in v:
                out_list.insert(-1 - loop_idx, "&" + k)
                out_list.insert(-1 - loop_idx, "&END " + k)
                iterdict(_v, out_list, loop_idx + 1)

        # if value is other type, e.g., int/float/str
        else:
            v = str(v)
            if k == "_":
                out_list[start_idx] = out_list[start_idx] + " " + v

            else:
                out_list.insert(-1 - loop_idx, k + " " + v)
                #out_list.insert(-1-loop_idx, v)

    return out_list


def dict2string(params:dict, delimiter:str="   ", aligned:bool=True, sort_keys:bool=False):
    """
    Returns a string representation of the dict.

    parameters:
        sort_keys (bool): Set to True to sort the dict parameters
            alphabetically. Defaults to False.
        pretty (bool): Set to True for pretty aligned output. Defaults
            to False.
    """
    keys = list(params.keys())
    if sort_keys:
        keys = sorted(keys)

    lines = []
    add = lines.append
    
    for k in keys:
        if isinstance(params[k], list):
            add([k, " ".join(list((str(i) for i in params[k])))])
        else:
            add([k, params[k]])
    
    if aligned:
        return str(tabulate([[line[0], "  ", line[1]] for line in lines],tablefmt="plain"))

    return "\n".join(list(delimiter.join(list((str(m) for m in line))) for line in lines))


def update_config_by_accuracy(accuracy:float, user_config:dict, iter_idx:int|None=None):

    accuracys = user_config["update_config"]["accuracy"]
    new_configs = user_config["update_config"]["new_config"]
    assert len(accuracys) == len(new_configs)

    if len(accuracys) != 0:
        criterion = accuracys[0]
        new_config = new_configs[0]
        if isinstance(new_config, str):
            new_config = load_config(new_config)

        if accuracy > float(criterion) and isinstance(new_config, dict):
            del user_config["update_config"]["accuracy"][0]
            del user_config["update_config"]["new_config"][0]
            if len(user_config["update_config"]["accuracy"]) == 0:
                del user_config["update_config"]

            update_dict(user_config, new_config)
            
            if iter_idx is None:
                with open(f"config-{criterion}.json", "w") as f:
                    json.dump(user_config, f, indent=4)

            else:
                with open(f"config-{criterion}-{iter_idx}.json", "w") as f:
                    json.dump(user_config, f, indent=4)

    return user_config


