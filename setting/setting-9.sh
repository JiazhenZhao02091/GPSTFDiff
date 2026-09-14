#!/bin/bash

# setting-9 改进版本
# 9.1: crop 反射率数据 patch_size_256_stride_256
# raw -> format -> crop_{}\_{}\_{}\_{} -> split_size_256_stride_256
# 9.2: crop 反射率数据 full
# raw -> format -> crop_{}\_{}\_{}\_{}

# 设置错误时退出
set -e

# 颜色输出函数
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

# 检查数据目录是否存在
check_data_dir() {
    if [ ! -d "data/spatio_temporal_fusion" ]; then
        print_error "数据目录 data/spatio_temporal_fusion 不存在！"
        exit 1
    fi
    print_info "数据目录检查通过"
}

# 检查Python环境
check_python_env() {
    if ! command -v python &> /dev/null; then
        print_error "Python 未安装或不在 PATH 中"
        exit 1
    fi
    print_info "Python 环境检查通过"
}

# 检查目标文件是否已存在
check_target_exists() {
    local target_pattern="$1"
    local step_name="$2"
    
    # 检查是否有任何匹配的目录存在且包含文件
    local found_files=false
    for path in data/spatio_temporal_fusion/*/; do
        if [ -d "$path" ]; then
            dataset_dir=$(basename "$path")
            # 构建完整的目标路径
            full_target_path="$path$target_pattern"
            if [ -d "$full_target_path" ] && [ "$(ls -A "$full_target_path" 2>/dev/null)" ]; then
                found_files=true
                break
            fi
        fi
    done
    
    if [ "$found_files" = true ]; then
        print_skip "$step_name 目标文件已存在，跳过执行"
        return 0  # 存在，跳过
    else
        return 1  # 不存在，需要执行
    fi
}

# 执行命令的函数
run_command() {
    local cmd="$1"
    local step_name="$2"
    local target_pattern="$3"
    
    # 检查目标文件是否已存在
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

# 主函数
main() {
    print_info "开始执行 setting-9 数据处理流程"
    
    # 检查环境
    check_data_dir
    check_python_env
    
    # 1. 生成format格式
    run_command \
        "python -m scripts.spatio_temparol_fusion.format.format --root_path data/spatio_temporal_fusion --src_data_prefix raw_data --tar_data_prefix format_data" \
        "生成format格式" \
        "public_processing_data/format_data"
    
    # 2. crop
    run_command \
        "python -m scripts.spatio_temparol_fusion.process.crop --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}" \
        "裁剪处理" \
        "public_processing_data/format_data/crop_*_*_*_*"
    
    # 3. 分割，否则就是full
    # split_size_256_stride_256
    run_command \
        "python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_{}_stride_{} --img_patch_size 128 --stride 128" \
        "图像分割处理" \
        "public_processing_data/format_data/crop_*_*_*_*/split_size_128_stride_128"
    
    # 4. dataset generation
    # 9.1:
    run_command \
        "python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_128_stride_128 --tar_data_prefix syy_setting-9-128 --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py" \
        "生成patch数据集" \
        "private_data/syy_setting-9"
    
    # 9.2:
    # run_command \
    #     "python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path /home/zhaojiazhen/stf/data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-9-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py" \
    #     "生成full数据集" \
    #     "private_data/syy_setting-9-full"
    
    print_info "setting-9 数据处理流程执行完成！"
}

# 执行主函数
main "$@" 
