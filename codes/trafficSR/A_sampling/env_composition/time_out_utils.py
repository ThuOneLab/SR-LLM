'''
Author: guozelin-ai 3190102461@zju.edu.cn
Date: 2025-02-17 23:00:54
LastEditors: guozelin-ai 3190102461@zju.edu.cn
LastEditTime: 2025-02-17 23:56:45
FilePath: \Symbolic_Regression_with_Large_Language_Models\codes\trafficSR\A_sampling\env_composition\time_out_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import multiprocessing
import queue
def worker(func, args, kwargs, queue):
    """
    工作函数，用于放入新进程中执行。
    :param func: 要执行的目标函数
    :param args: 目标函数的位置参数
    :param kwargs: 目标函数的关键字参数
    :param queue: 用于返回结果的队列
    """
    try:
        result = func(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def run_with_timeout_multiprocessing(func, args=(), kwargs={}, timeout_duration=5):
    result_queue = multiprocessing.Queue()

    # # 定义实际的工作函数，用于放入新进程中执行
    # def worker(func, args, kwargs, queue):
    #     result = func(*args, **kwargs)
    #     queue.put(result)

    # 创建进程
    p = multiprocessing.Process(target=worker, args=(func, args, kwargs, result_queue))
    
    # 启动进程
    p.start()
    
    result = None
    try:
        # 等待指定的秒数以获取结果
        result = result_queue.get(timeout=timeout_duration)
    except queue.Empty:
        print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
    finally:
        if p.is_alive():
            print("Terminating process...")
            p.terminate()  # 强制终止进程
            p.join()  # 确保进程已结束
        
    return result if result is not None else None

# for test
def timeout_test():
    def long_running_task(duration):
        import time
        time.sleep(duration)
        return "Task Completed"

    # Run the function with a timeout mechanism
    result = run_with_timeout_multiprocessing(long_running_task, args=(2,), timeout_duration=3)
    print(result)