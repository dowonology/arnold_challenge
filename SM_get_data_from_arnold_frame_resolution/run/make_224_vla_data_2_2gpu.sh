#!/bin/bash
# update_water.sh: 인자로 GPU 번호만 받도록 수정

frame_num=$1

# 나머지 변수는 고정값으로 설정
make_name="Make_224_vla_data_frame_${frame_num}"
only_check=false


# Task 배열 정의
tasks=('open_cabinet' 'close_cabinet' 'close_drawer' 'transfer_water')
#tasks=('pour_water' 'transfer_water')
#tasks=('transfer_water')

# 탐색할 디렉토리 경로
directory="/vagrant/challenge_data_train"

# output_root_base 정의 (time_step 관련 부분 제거)
if [ "$only_check" = true ]; then
    output_root_base="/root/arnold/VLA_224_DATA/${make_name}_only_check"
else
    output_root_base="/root/arnold/VLA_224_DATA/${make_name}_NEW"
fi

# 디렉토리 확인
echo "Searching in directory: $directory"

# 모든 task 폴더 탐색
for task in "${tasks[@]}"; do
    task_dir="${directory}/${task}"
    
    # task 디렉토리가 존재하는지 확인
    if [ -d "$task_dir" ]; then
        echo "Processing task: $task"

        # 각 task 세트를 1번 반복 (필요 시 반복 횟수 조정)
        for i in {1..1}; do
            # 각 iteration에 맞는 output_root 설정
            iteration_output_root="${output_root_base}/output_iteration_${i}"
            output_root="${iteration_output_root}/${task}"
            mkdir -p "$output_root"
            
            echo "Running task: $task with checkpoint_file: $checkpoint_file and output_root: $output_root"
            
            /isaac-sim/python.sh "/root/arnold/make_224_vla_data_2.py" \
                task="$task" \
                use_gt="[1,1]" \
                visualize=0 \
                record=False \
                output_root="$output_root" \
                data_root="$directory" \
                eval_splits="[train]" \
                output_folder="$output_root" \
                only_check="$only_check" \
                make_data=True \
		make_name="$make_name" \
		target_frame_num="$frame_num"
        done
    else
        echo "Task directory '$task_dir' does not exist, skipping."
    fi
done

# 모든 eval_headless.py 실행이 종료되면 get_scores.py 실행
#echo "All evaluations completed. Running get_scores.py..."
#/isaac-sim/python.sh /root/arnold/get_scores.py "$output_root_base"

echo "Script completed."

