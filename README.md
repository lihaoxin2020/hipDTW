# hipDTW

## Build

Check out `./build`. 

Run `./build cpuDTW.cpp` or `./build hipDTW.cu` for cpu or Hip implementation respectively. 

## Test

`./reference.bin` contains 100k normalized values as test reference sequence. 

`./query.bin` contains 512 * 2k = 1024k random values of range [58, 120] as test queries. 

`./test-small.bin` short version of `./query.bin`.

Test files are written and read by helper functions in `./utils.h`.

To run tests, `sbatch ./run_tests <built_executable> <test_file>`
