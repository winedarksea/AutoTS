"""CPU counter for multiprocesing."""


def cpu_count(modifier: float = 1):
    """Find available CPU count, running on both Windows/Linux.

    Attempts to be very conservative:
        * Remove Intel Hyperthreading logical cores
        * Find max cores allowed to the process, if less than machine has total

    Runs best with psutil installed, fallsback to mkl, then os core count/2

    Args:
        modifier (float): multiple CPU count by this value
    """
    import os

    # your basic cpu count, includes logical cores and all of machine
    num_cores = os.cpu_count()
    if num_cores is None:
        num_cores = -1

    # includes logical cores, and counts only cores available to task
    try:
        import psutil

        available_cores = len(psutil.Process().cpu_affinity())
    except Exception:
        # this only works on UNIX I believe
        try:
            available_cores = len(os.sched_getaffinity(0))
        except Exception:
            available_cores = -1

    # only physical cores, includes all available to machine
    try:
        import psutil

        ps_cores = psutil.cpu_count(logical=False)
    except Exception:
        try:
            import mkl

            ps_cores = int(mkl.get_max_threads())
        except Exception:
            ps_cores = int(num_cores / 2)

    core_list = [num_cores, available_cores, ps_cores]
    core_list = [x for x in core_list if x > 0]
    if core_list:
        core_count = min(core_list)
    else:
        core_count = 1
    if modifier != 1:
        core_count = int(modifier * core_count)
    core_count = 1 if core_count < 1 else core_count
    return core_count


def set_n_jobs(n_jobs, verbose=0):
    if n_jobs is None:
        return None
    frac_flag = False
    if isinstance(n_jobs, float):
        frac_flag = n_jobs < 1 and n_jobs > 0
    if n_jobs == 'auto' or frac_flag or n_jobs == -1:
        if frac_flag:
            n_jobs = cpu_count(modifier=n_jobs)
        else:
            n_jobs = cpu_count(modifier=0.75)
        if verbose > 0:
            print(f"Using {n_jobs} cpus for n_jobs.")
    elif str(n_jobs).isdigit():
        n_jobs = int(n_jobs)
    elif n_jobs < 0:
        core_count = cpu_count(modifier=1) + 1 + n_jobs
        n_jobs = core_count if core_count > 1 else 1
    elif isinstance(n_jobs, (float, int)):
        pass
    else:
        raise ValueError("n_jobs must be 'auto' or integer")
    if n_jobs <= 0:
        n_jobs = 1
    return int(n_jobs)
