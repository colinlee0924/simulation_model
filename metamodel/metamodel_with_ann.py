import os

input_dir = os.getcwd() + '/input_data/'
out_dir = os.getcwd() + '/output_result/'
log_dir = os.getcwd() + '/runs/'
dir_list = [input_dir, out_dir, log_dir]

for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)
