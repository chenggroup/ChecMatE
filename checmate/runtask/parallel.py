from multiprocessing import Process, Pool
from multiprocessing.dummy import Pool as ThreadPool
    

# a few tasks for cpu
def parallel_by_process(task_list:list, **kwargs):
    
    multi_processes = []
    for task in task_list:
        process = Process(target = task, **kwargs)
        process.start()
        multi_processes.append(process)

    for p in multi_processes:
        p.join()

# lots of tasks for cpu
def parallel_by_pool(func, iterator, n_process:int=4, whether_to_return:bool=False):

    pool = Pool(processes=n_process) if n_process else Pool()
    results = pool.map(func, iterator)
    pool.close()
    pool.join()

    if whether_to_return:
        return results

# lots of tasks for io
def parallel_by_threadpool(func, iterator, n_process:int=4, whether_to_return:bool=False):

    pool = ThreadPool(processes=n_process) if n_process else ThreadPool()
    results = pool.map(func, iterator)
    pool.close()
    pool.join()

    if whether_to_return:
        return results
