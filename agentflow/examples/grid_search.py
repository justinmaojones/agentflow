import os
import random


def sample(v):
    i = random.randint(0, len(v) - 1)
    return v[i]


def build_mkfile(tasks):
    task_names = []
    task_str = []
    for i, task in enumerate(tasks):
        task_names.append("TASK%d" % i)
        task_str.append("%s:" % task_names[-1])
        task_str.append("\tsleep %d;" % i)
        task_str.append("\t%s;" % task.strip("\n"))
    output = []
    output.append("all: " + " ".join(task_names))
    output.extend(task_str)
    return "\n".join(output)


def run_grid_search(num_trials, num_workers, run_file, save_dir, run_kwargs):
    """
    Runs a parallelized random grid search on a python file that takes
    command line kwargs as input.  Uses make, and thus requires it to
    be installed.

    Parameters
    ----------
    num_trials : integer
        Number of random trials to run.
    num_workers : integer
        Number of parallel workers.
    run_file : string
        Python file to execute
    save_dir : string
        Directory to save results to.
    run_kwargs : dict
        A dictionary where each key is a command-line option for the run file,
        and each value is a list of possible inputs to randomly select from.

    Examples
    --------
    The following code will run 20 trials with 4 workers. In each trial, a
    learning rate and hidden dim will be randomly selected from the values
    provided in `run_kwargs`.

    ```
    run_kwargs = {
        'learning_rate': [0.0001, 0.001],
        'hidden_dims': [32, 64],
    }
    run_grid_search(
        num_trials = 20,
        num_workers = 4,
        run_file = 'agentflow/examples/cartpole_ddpg.py',
        save_dir = 'results/cartpole_ddpg_grid_search',
        run_kwargs = run_kwargs,
    )
    ```
    """

    # build tasks
    tasks = []
    for t in range(num_trials):

        cmd = ["python %s" % run_file]
        for k in run_kwargs:
            v = sample(run_kwargs[k])
            cmd.append("--%s=%s" % (k, str(v)))
        cmd = " ".join(cmd)
        tasks.append(cmd)

    # create directory (if doesn't already exist)
    os.system("mkdir -p %s" % save_dir)

    # create a make file with tasks
    filepath = os.path.join(save_dir, "mkfile_experiments")
    with open(filepath, "w") as f:
        f.write(build_mkfile(tasks))

    # execute
    os.system("make -j %d -f %s" % (num_workers, filepath))
