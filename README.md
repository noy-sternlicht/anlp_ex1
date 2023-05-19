# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on
the SST2 dataset.

# Quick Start

1. Clone the repository:

   ``` git clone https://github.com/noy-sternlicht/anlp_ex1.git ```
   
2. Enter the repository:

   ```cd anlp_ex1```

3. Create a virtual environment:

   ``` python3 -m venv myenv ```
4. Activate the virtual environment:

   ``` source myenv/bin/activate ```
5. Install requirements:

   ``` pip install -r requirements.txt ```

Alternatively, on cluster, you can run the build_venv.sh script, which will do 3..5 for you:

```chmod +x ./build_env.sh```

```sbatch ./build_env.sh ```

# Fine-Tune and Predict on Test Set

Run:

``` python ex1.py <number of seeds>  <number of training samples> <number of validation samples> <number of prediction samples> ```

Alternatively, on cluster, you can run:

```chmod +x ./run.sh```

```sbatch ./run.sh ```

It will run ex1.py on an a10 GPU with seeds 1..3, on all training, validation and prediction sets.

Generated files are res.txt, showing model's performance, training time and prediction time, and prediction.txt,
containing prediction results for all test samples.
