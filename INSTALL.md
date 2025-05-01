# pmg-dolfinx

P-multigrid on GPUs using dolfinx

## LUMI [MI250X]

Install the spack environment in `hypre-rocm-lumi.yaml`
```
spack env create hypre-rocm hypre-rocm-lumi.yaml
spack env activate hypre-rocm
```

N.B. to install fenics-basix, you might need to patch spack slightly:
`spack edit fenics-basix` and add the following:
```
def cmake_args(self):
    options = ["-DBLAS_LIBRARIES=" + self.spec["blas"].libs.joined(),
          "-DLAPACK_LIBRARIES=" + self.spec["blas"].libs.joined()]
    return options
```

Once `spack install` has completed successfully, you can build the
examples in the `examples` folder using cmake.
```
# Setup required for "build" on login node
module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
source spack/share/spack/setup-env.sh
spack env activate hypre-rocm
spack load cmake

# Build an example
cd pmg-example
mkdir build
cd build
cmake -Damd=ON ..
make
```

Get a GPU node:

```
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=01:0:00 --partition=dev-g --account=ACCOUNT
export MPICH_GPU_SUPPORT_ENABLED=1
srun --ntasks=1 ./pmg --ndofs=50000
```

There are some instructions on running on GPU on the LUMI documentation,
especially about [selecting GPU/CPU
affinity](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/).
