import pytest
from pathlib import Path
from shutil import rmtree
from ase.io import read
from checmate.pretask.inputs import *



structure = read("../beta-Ga2O3.cif")
type_map = ["Ga", "In", "O"]
inputClass = [
    ("VaspRelaxInput(structure)", ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]),
    ("VaspStaticInput(structure)", ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]),
    ("LaspInput(structure)", ["lasp.in", "lasp.str"]),
    ("LammpsInput(structure, {'type_map': type_map, 'models': ['frozen_model1.pb']})", ["in.simple", "data.simple"]),
    ("DPTrainInput(type_map, [])", ["input.json"]),
    ("RunInput()", ["machine.json"]),
    ("Cp2kRelaxInput(structure)", ["coord.xyz", "cp2k.inp"]),
    ("Cp2kStaticInput(structure)", ["coord.xyz", "cp2k.inp"])
]


@pytest.fixture(params=inputClass)
def class_input_fixture(request):

    temp_dir = Path("temp_dir")
    if temp_dir.exists():
        rmtree(temp_dir)
    yield request.param
    rmtree(temp_dir)


def test_wirte_input(class_input_fixture):
    
    runclass = eval(class_input_fixture[0])
    runclass.write_input("temp_dir")

    files_list = [i.name for i in Path("temp_dir").glob("*")]
    assert sorted(class_input_fixture[1]) == sorted(files_list)


def method_input(method):
    
    config = {'test': 123}
    return eval(method)


def test_method_input():
    
    inputmethod = [
    ("LaspInput(structure, user_config=config)", "laspin"),
    ("DPTrainInput(type_map, [], user_config=config)", "dpinput"),
    ("RunInput(machine=config, resources=config)", "machine_and_resources"),
    ("Cp2kRelaxInput(structure, user_config=config)", "cp2kinp")
]

    runclass = ('.'.join(i) for i in inputmethod)
    result = map(method_input, runclass)

    for i in result:
        if type(i) == dict:
            assert 'test' in i
        elif type(i) == tuple:
            assert 'test' in i[0] and 'test' in i[1]


def test_write_input_vasp():

    tmp = VaspRelaxInput(structure, user_config={"incar":{}, "kpoints":{123}, "potcar":{}})
    assert tmp.user_kpoints_settings == {123}


def test_vasp_pre():

    files_to_transfer = VaspStaticInput.from_pre_calc(".", {"setting.json": None})
    dirname = Path(".").absolute()
    assert files_to_transfer == {"CHGCAR": str(dirname/"CHGCAR"), "setting.json": str(dirname/"setting.json")}
