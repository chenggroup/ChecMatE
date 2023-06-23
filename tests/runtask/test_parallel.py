import os, time
from checmate.runtask.parallel import *



def fun1(a, b):

    time.sleep(1)
    return a**3 + b**3



def fun2(z):

    return fun1(z[0], z[1])



def fun3(txt):

    time.sleep(1)
    with open("tmp.txt", "w") as f:
        f.write(txt*1000)



def test_parallel_by_process():
    
    t0 = time.time()
    for i in range(10):
        result = fun2((1000,2000))
    t1 = time.time() - t0

    t0 = time.time()
    task_list = [fun2]*10
    result = parallel_by_process(task_list=task_list, args=(1000,2000))
    t2 = time.time() - t0

    print(f"Process: serial(t1):{t1}; parallel(t2):{t2}")
    assert t1 > t2



def test_parallel_by_pool():

    numb = ((1000, 2000) for i in range(10))

    t0 = time.time()
    for i in numb:
        result = fun2(i)
    t1 = time.time() - t0

    t0 = time.time()
    result = parallel_by_pool(func=fun2, iterator=numb)
    t2 = time.time() - t0

    print(f"Pool: serial(t1):{t1}; parallel(t2):{t2}")
    assert t1 > t2



def test_parallel_by_threadpool():

    txt = ('a'*i for i in range(100,110))

    t0 = time.time()
    for i in txt:
        result = fun3(i)
    t1 = time.time() - t0

    t0 = time.time()
    result = parallel_by_threadpool(func=fun3, iterator=txt)
    t2 = time.time() - t0

    print(f"ThreadPool: serial(t1):{t1}; parallel(t2):{t2}")
    assert t1 > t2
    os.remove('tmp.txt')
