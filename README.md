## Soft error test using CUDA

The program is developed to test the soft errors on GPU using CUDA

### Test Design
1. Simply write and read operation on GPU
2. Alternating pattern of static and dynamic blocks to write and verify data
3. Write and retain the data for long time
4. bit reverse operations

### How to run the program
Use the`run.sh` to compile the c++ files and run the `./cuda_application`

Arguments:
- `--gpu_id=`: The device that runs the program. Default device is GPU 0.
- `--total_data=`: The size of memory in MB being used. Default size is 8192MB.
- `-n=`: The number of iterations of all the programs. Default number is 1 iteration.

### AWS Deployment
1. Download the AWS client
```
sudo apt update
sudo apt install awscli -y
```
2. Config AWS user using access key and secret key
```
aws configure
```
3. Use AWS s3 to upload the program
```
aws s3 cp soft-error-tester s3://bucket-name/ --recursive
```
