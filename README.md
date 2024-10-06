# Estimating Generalization Performance Along the Trajectory of Proximal SGD in Robust Regression

This repository contains the code to reproduce the result in the following paper:
* Estimating Generalization Performance Along the Trajectory of Proximal SGD in Robust Regression. [[arXiv]](https://arxiv.org/pdf/2410.02629)

## File description:

* `run_gpu_GD.py`: Executes the experiments of GD and proximal GD. It generates the left panel of Figure 1 and Figure 2.
* `run_gpu_SGD.py`: Executes the experiments of SGD and proximal SGD. It generates the right panel of Figure 1 and Figure 2.
* `SGD_varing_step_siz.ipynb`: Generates Figure 3.
* `SGD_suboptimal_estimate.ipynb`: Generates Figure 4.

## Detailed executing steps:

1. Identify a computing environment equipped with a GPU, such as a high-performance computing cluster you have access to. 
1. Manually adjust the `dim_set` and `loss_set` in the script `run_gpu_GD.py` and `run_gpu_SGD.py` to specify the simulation settings of $(n,p,T)$ and the loss functions. For each combination, 
    * Run the script `run_gpu_GD.py` to generate all the figures on GD and proximal GD in Figures 1-2. 
    * Run the script `run_gpu_SGD.py` to generate all the figures on SGD and proximal SGD in Figures 1-2.
1. Run the Jupyter notebook `SGD_varing_step_siz.ipynb` with different choice of `eta_set` in the code (corresponding to different choices of step size $\eta_t$ in Figure 3). 
This generates Figure 3.
1. Run the Jupyter notebook `SGD_suboptimal_estimate.ipynb` with different choice of `loss_set` in the code (corresponding to Huber loss and pseudo Huber loss in Figure 4). 
This generates the Figure 4.

## Hardware and computing times:
For step 2, we used a computing environment equipped with the GPU card NVIDIA A100-PCIE-40GB. 
It takes less than 40 seconds for each experiment, and 100 repetitions of the experiments can be completed in less than 90 minutes.

For steps 3 and 4, we used a local machine (MacBook Pro 2021) equipped with an Apple M1 Pro chip and 16 GB of memory. Step 3 and step 4 take 30 seconds and 90 seconds, respectively, to complete for each setting.

## Citation:
```
@article{tan2024estimating,
  title={Estimating Generalization Performance Along the Trajectory of Proximal SGD in Robust Regression},
  author={Tan, Kai and Bellec, Pierre C},
  journal={arXiv preprint arXiv:2410.02629}, 
  year={2024}
}
```