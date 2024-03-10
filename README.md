# hipDTW

## Test

`./reference.txt` contains 100k normalized values as test reference sequence. 

`./query.txt` contains 512 x 2k = 1024k random values of range [58, 120] as test queries. (Note that first 50 batches are subsequences from reference, and therefore low sDTW scores are expected) 

We provide 2 executables, both contain exact same kernel implementation, but used for different tests:
- `./hipDTW` is used for correctness check, which will print out mean, std_dev, and the lowest score for each batch. You can check correctness by comparing output prints with `cpusDTW.txt`, which contains scores calculated by CPU. We provide an example output `./correctness-test.out`. 
- `./hipDTW_test` is used for throughput check, which will NOT print any kernal output but only tests for throughput. Example output `./throughput-test.out`. 

To run tests, `sbatch ./run_tests <built_executable> <query_file> <reference_file>`

## Build

Check out `./build`. 

Run `./build cpuDTW.cpp` or `./build hipDTW.cu` or `./build hipDTW_test.cu` for cpu or Hip implementation respectively. 
