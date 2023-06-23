from pathlib import Path
from checmate.runtask.run import RunTask
from dpdispatcher.submission import Task



def test_run_task(make_temp_dir):
    
    temp_dir = Path("temp_dir")

    runtask = RunTask(
        output_dir=temp_dir,
        dpdispatcher_config={
                "forward_common_files":["./CONTCAR"],
                "backward_common_files":["test.back"]
        }
    )
    
    Path.mkdir(temp_dir/"task.01")
    with open(str(temp_dir/"task.01"/"test.in"), "w") as f:
        f.write("test")

    task1 = Task(
        command="cat test.in > test.out && cat ../CONTCAR > ../test.back",
        task_work_path="task.01",
        forward_files=["test.in"],
        backward_files=["test.out", "task.err", "task.log"],
        errlog="task.err",
        outlog="task.log"
    )

    Path.mkdir(temp_dir/"task.02")

    task2 = Task(
        command="touch test1.arc",
        task_work_path="task.02",
        forward_files=[],
        backward_files=["test1.arc", "task.err", "task.log"],
        errlog="task.err",
        outlog="task.log"
    )
    
    submission = runtask.submit_task(task_list=[task1, task2], whether_to_run=False)

    assert Path("temp_dir/machine.json").is_file()
    assert Path("temp_dir/CONTCAR").is_symlink()
    assert submission["work_base"] == "temp_dir"
    assert submission["forward_common_files"] == ["CONTCAR"]
    assert submission["backward_common_files"] == ["test.back"]

    submission.run_submission()
    
    files_list = [i.name for i in Path(temp_dir/"task.01").glob("t*")]
    assert sorted(["test.in", "test.out", "task.err", "task.log"]) == sorted(files_list)

    with open(str(temp_dir/"task.01"/"test.in"), "r") as fin, open(str(temp_dir/"task.01"/"test.out"),"r") as fout:
        assert fin.read() == fout.read()
    
    assert Path(temp_dir/"test.back").is_file()
    with open(str(temp_dir/"CONTCAR"), "r") as fin, open(str(temp_dir/"test.back"), "r") as fout:
        assert fin.read() == fout.read()


