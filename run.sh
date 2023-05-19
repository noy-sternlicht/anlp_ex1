#!/bin/bash -x
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a10:1
#SBATCH --mem-per-cpu=70g

activate() {
  . $PWD/myenv/bin/activate
}

activate

python3 ex1.py 3 -1 -1 -1
