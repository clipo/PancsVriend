#!/bin/bash
#SBATCH --job-name=llm_schelling
#SBATCH --output=spie/llm_schelling_output.txt
#SBATCH --error=spie/llm_schelling_error.txt
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --time=122:00:00

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi

export OMP_NUM_THREADS=$omp_threads

python3 comprehensive_comparison_study.py 
