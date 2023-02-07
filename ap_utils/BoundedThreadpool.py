from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import BoundedSemaphore

class BoundedExecutor:
    """
    BoundedExecutor behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.

    by Frank Cleary

    https://www.bettercodebytes.com/theadpoolexecutor-with-a-bounded-queue-in-python

    Parameters:
        * bound: Integer - the maximum number of items in the work queue
        * max_workers: Integer - the size of the thread pool

    Methods:
        * submit
        * shutdown
    """

    def __init__(self, bound, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound + max_workers)

    def submit(self, fn, *args, **kwargs):
        """
        Submit a task to the ThreadPoolExecutor.
        See concurrent.futures.Executor#submit
        """
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

    def shutdown(self, wait=True):
        """
        Shutdown the ThreadPoolExecutor, optionally
        wait for all tasks to complete.
        See concurrent.futures.Executor#shutdown
        """
        self.executor.shutdown(wait)
