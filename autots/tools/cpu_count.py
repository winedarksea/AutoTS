"""CPU counter for multiprocesing."""


def cpu_count():
    """Find available CPU count, running on both Windows/Linux.

    Attempts to be very conservative:
        * Remove Intel Hyperthreading logical cores
        * Find max cores allowed to the process, if less than machine has total

    Runs best with psutil installed, fallsback to mkl, then os core count/2
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
    return core_count
