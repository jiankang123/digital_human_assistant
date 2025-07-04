import sys
import random

input_file_path = sys.argv[1]
select_file_path = sys.argv[2]
left_file_path = sys.argv[3]
rate = float(sys.argv[4])
all_file_list = []

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    try:
        for line in input_file.readlines():
            all_file_list.append(line)
    except:
        print(line)
random.shuffle(all_file_list)
select_file_list = random.sample(all_file_list, int(len(all_file_list) * rate))
left_file_list = list(set(all_file_list) - set(select_file_list))
with open(select_file_path, 'w', encoding='utf-8') as select_file:
    select_file.write(''.join(select_file_list))
with open(left_file_path, 'w', encoding='utf-8') as left_file:
    left_file.write(''.join(left_file_list))