import yaml
import argparse
import subprocess
import active_dynamicmemory.runutils as rutils


def train_params(params, remote=False, runs=None, jobarray=False):
    if remote:
        with open('training_configs/slurm_config.yml') as s:
            sparams = yaml.load(s, Loader=yaml.FullLoader)

    if jobarray:
        sparams['qos'] = 'jobarray'

    if runs is None:
        if remote:
            print('Scheduling job to CIR cluster...')
            cmd = ['srun', 'python', '-m', 'active_dynamicmemory.runutils.trained_model',
                   params['trainparams'], params['settings']]
            subprocess.run(cmd, check=True)
        else:
            model, logs, df_mem, exp_name = rutils.trained_model(params['trainparams'], params['settings'])
            print('Successfully trained model', exp_name)
            print('Metrics file path:', f"{params['settings']['LOGGING_DIR']}{exp_name}/version_0/metrics.csv")
    else:
        for i in range(runs):
            params['trainparams']['seed'] = i + 1
            params['trainparams']['run_postfix'] = i + 1
            if remote:
                print('Scheduling job to CIR cluster...')
                cmd = ['srun', 'python', '-m', 'active_dynamicmemory.runutils.trained_model',
                       params['trainparams'], params['settings']]
                subprocess.run(cmd, check=True)
            else:
                model, logs, df_mem, exp_name = rutils.trained_model(params['trainparams'], params['settings'])
                print('Successfully trained model', exp_name)
                print('Metrics file path:', f"{params['settings']['LOGGING_DIR']}{exp_name}/version_0/metrics.csv")


def train_config(configfile, remote=False, runs=None, jobarray=False):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    train_params(params, remote=remote, runs=runs, jobarray=jobarray)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a training with the dynamic memory framework.')
    parser.add_argument('--config', type=str, help='path to a config file (yml) to run')
    parser.add_argument('-s', '--slurm', action='store_true', help='run on CIR slurm cluster')
    parser.add_argument('-r', '--runs', type=int, help='number of runs with the config')
    args = parser.parse_args()


    train_config(args.config, remote=args.slurm, runs=args.runs)