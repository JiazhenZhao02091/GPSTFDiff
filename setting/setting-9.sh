#!/bin/bash








set -e


print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_skip() {
    echo -e "\033[36m[SKIP]\033[0m $1"
}


check_data_dir() {
    if [ ! -d "data/spatio_temporal_fusion" ]; then
        print_error "数据目录 data/spatio_temporal_fusion 不存在！"
        exit 1
    fi
    print_info "数据目录检查通过"
}


check_python_env() {
    if ! command -v python &> /dev/null; then
        print_error "Python 未安装或不在 PATH 中"
        exit 1
    fi
    print_info "Python 环境检查通过"
}


check_target_exists() {
    local target_pattern="$1"
    local step_name="$2"


    local found_files=false
    for path in data/spatio_temporal_fusion/*/; do
        if [ -d "$path" ]; then
            dataset_dir=$(basename "$path")

            full_target_path="$path$target_pattern"
            if [ -d "$full_target_path" ] && [ "$(ls -A "$full_target_path" 2>/dev/null)" ]; then
                found_files=true
                break
            fi
        fi
    done

    if [ "$found_files" = true ]; then
        print_skip "$step_name 目标文件已存在，跳过执行"
        return 0
    else
        return 1
    fi
}


run_command() {
    local cmd="$1"
    local step_name="$2"
    local target_pattern="$3"


    if check_target_exists "$target_pattern" "$step_name"; then
        return 0
    fi

    print_info "开始执行: $step_name"
    echo "命令: $cmd"

    if eval "$cmd"; then
        print_info "$step_name 执行成功"
    else
        print_error "$step_name 执行失败"
        exit 1
    fi
    echo "----------------------------------------"
}


main() {
    print_info "开始执行 setting-9 数据处理流程"


    check_data_dir
    check_python_env


    run_command \
        "python -m scripts.spatio_temparol_fusion.format.format --root_path data/spatio_temporal_fusion --src_data_prefix raw_data --tar_data_prefix format_data" \
        "生成format格式" \
        "public_processing_data/format_data"


    run_command \
        "python -m scripts.spatio_temparol_fusion.process.crop --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}" \
        "裁剪处理" \
        "public_processing_data/format_data/crop_*_*_*_*"



    run_command \
        "python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_{}_stride_{} --img_patch_size 128 --stride 128" \
        "图像分割处理" \
        "public_processing_data/format_data/crop_*_*_*_*/split_size_128_stride_128"



    run_command \
        "python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_128_stride_128 --tar_data_prefix syy_setting-9-128 --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py" \
        "生成patch数据集" \
        "private_data/syy_setting-9"







    print_info "setting-9 数据处理流程执行完成！"
}


main "$@"
