import os
import subprocess
import sys
import time
import argparse

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Launch PrefPalette training pipeline with YAML config")
    parser.add_argument('--master_port', type=int, default=29500)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--train_yaml_path', type=str, required=True, help='Path to YAML config file or directory')
    parser.add_argument('--train_overrides', type=str, default=None, help='Pipe-separated key=value overrides (e.g., "key1=val1|key2=val2")')
    parser.add_argument('--train_module', type=str, default="openrlhf.cli.train_rm", help='Training module to run')
    return parser.parse_args()


def launch_training(args, config):
    avail_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', ",".join(map(str, range(avail_gpus))))

    if avail_gpus < args.num_gpus:
        raise ValueError(f"Requested {args.num_gpus} GPUs but only {avail_gpus} available")
    elif avail_gpus > args.num_gpus:
        cuda_visible_devices = ",".join(cuda_visible_devices.split(",")[:args.num_gpus])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    if args.train_overrides:
        for override in args.train_overrides.split("|"):
            if '=' in override:
                key, value = override.split('=', 1)
                if value.lower() == "true":
                    config[key] = True
                elif value.lower() == "false" and key in config:
                    del config[key]
                else:
                    config[key] = value
            else:
                config[override] = True

    training_args = []
    for k, v in config.items():
        if v is False:
            continue
        training_args.append(f"--{k}")
        if v is True or v is None:
            continue
        training_args.append(str(v))

    saved_cwd = os.getcwd()
    saved_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    if saved_cuda:
        del os.environ['CUDA_VISIBLE_DEVICES']

    training_command = ['deepspeed', '--master_port', str(args.master_port)]
    if saved_cuda:
        training_command.extend(['--include', 'localhost:' + saved_cuda])
    training_command += ['--module', args.train_module] + training_args

    print(f"Running: {' '.join(training_command)}")
    subprocess.run(training_command, env=os.environ, stdout=sys.stdout, stderr=subprocess.STDOUT)

    if saved_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = saved_cuda
    os.chdir(saved_cwd)


def main():
    args = parse_args()

    if os.path.isdir(args.train_yaml_path):
        yaml_files = sorted([os.path.join(args.train_yaml_path, f) for f in os.listdir(args.train_yaml_path) if f.endswith('.yaml')])
    else:
        yaml_files = [args.train_yaml_path]

    print(f"Found {len(yaml_files)} training configs")

    for yaml_path in yaml_files:
        print(f"\n{'='*60}")
        print(f"Training with config: {yaml_path}")
        print(f"{'='*60}")
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        start = time.time()
        launch_training(args, config)
        print(f"Training took {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
