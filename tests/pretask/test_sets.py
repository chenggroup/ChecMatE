from os import remove
from ase.io import read
from checmate.pretask.inputs import DPTrainInput, Cp2kStaticInput
from checmate.pretask.sets import load_config, update_dict, update_config_by_accuracy

structure = read("../beta-Ga2O3.cif")

def test_load_config():

    assert type(load_config("setting.json")) == dict


def test_update_dict1():

    dict_1 = DPTrainInput([],[]).dpinput
    txt = {'test':'123'}

    update_dict(dict_1, txt)
 
    assert 'test' in dict_1


def test_update_dict2():

    dict_1 = Cp2kStaticInput(structure).cp2kinp
    dict_2 = DPTrainInput([],[]).dpinput
    assert update_dict(dict_1, dict_2) == update_dict(dict_2, dict_1)


def test_update_config_by_accuracy():

    user_config = load_config("setting.json")
    user_config = update_config_by_accuracy(1, user_config)
    assert user_config == {"test":456}
    remove("config-0.json")



    
