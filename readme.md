
## Environment Preparation
### Ours Environments
```
Linux user-ThinkStation-P520 6.2.0-34-generic #34~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC
Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz
65:00.0 VGA compatible controller: NVIDIA Corporation Device 26b1 (rev a1)
```
### Docker setup
```
    docker build -t mlirbenchmark -f env/Dockerfile.nv .

    docker run -it --cap-add=SYS_PTRACE --gpus all --name mlirbenchmark -v ~/mlc/mlir:/root/mlir --shm-size="32g" --privileged=true mlirbenchmark /bin/bash

    #docker start mlirbenchmark

    #docker exec -it mlirbenchmark /bin/bash
```
### Install and reproduce
```
cd mlir
chmod 777 env/setup.sh
cd env
#安装nvidia profiler工具
dpkg -i nsight-systems-2023.3.1_2023.3.1.92-1_amd64.deb
./setup.sh
cd baseline
chmod 777 *.sh
./run.sh
```