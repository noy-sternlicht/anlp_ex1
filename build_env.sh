#!/bin/bash -x
#SBATCH --time=3:00:00
#SBATCH -c10
#SBATCH --mem=50g

PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv/bin/activate
}

virtualenv myenv
activate
set_env_vars

# Install packages:
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install -r requirements.txt