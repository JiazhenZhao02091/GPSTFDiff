function run_fitfc(base_dir, dataset_name, W, result_dir, data_range)
% 主函数必须和文件名一致
% matlab -batch "run_fitfc('/path/to/config2/full', 'CIA_config2', 7)"
% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full', 'CIA', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/CIA/full')"
% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch', 'CIA', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/CIA/patch')"



% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full', 'LGC', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/LGC/full')"
% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/patch', 'LGC', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/LGC/patch')"



% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/patch', 'ML', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/ML/patch', '[0, 1]')"
% matlab -batch "run_fitfc('/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full', 'ML', 15, '/home/zhaojiazhen/workspace/STF/STF_matlab/results/ML/full', '[0, 1]')"


% 使用 nargin 判断是否传入了参数。如果没有传入，则使用默认值。
% 这样既支持命令行传参，也支持在 MATLAB 界面直接按 F5 运行默认配置。

if nargin < 1 || isempty(base_dir)
    base_dir = '/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full';
end

if nargin < 2 || isempty(dataset_name)
    dataset_name = 'CIA'; 
end

if nargin < 3 || isempty(W)
    W = 15;
else
    % 如果是从 shell -batch 传进来的，数字会被当成字符串，需要转换
    if ischar(W) || isstring(W)
        W = str2double(W);
    end
end

if nargin < 4 || isempty(result_dir)
    result_dir = '/home/zhaojiazhen/workspace/STF/STF_matlab/results';
end

if nargin < 5 || isempty(data_range)
    data_range = [0, 10000]; % 默认值
else
    % 如果是从终端 batch 传进来的字符串 "[0, 1]"，需要转化
    if ischar(data_range) || isstring(data_range)
        data_range = str2num(data_range); 
    end
end

% 固定参数区
% data_range = [0, 1]; % 数据范围参数
B1 = 3; % 红波段
B2 = 4; % 近红外波段
s  = 1; % 缩放比，尺寸相同所以为 1
% =======================================================


% ==================== 获取所有待处理的图片 =================
% 查找 Landsat_01 目录下所有的 .tif 文件
landsat_dir = fullfile(base_dir, 'Landsat_01');
tif_files = dir(fullfile(landsat_dir, '*.tif'));

fprintf('共找到 %d 个文件需要处理。\n', length(tif_files));

% ==================== 循环处理每一组 =======================
for file_idx = 1:length(tif_files)
    
    img_name_landsat = tif_files(file_idx).name;
    % 假设文件名为 Group_01_L_.tif，利用替换获取对应 MODIS 名称
    img_name_modis   = strrep(img_name_landsat, '_L_', '_M_'); 
    
    % 提取基础名称（去掉后缀和_L_）
    base_name = strrep(img_name_landsat, '.tif', ''); 
    base_name = strrep(base_name, '_L_', '_'); % 变成 Group_01_0_256_0_256
    
    % 提取组名作为输出前缀，如 "Group_01"
    parts = strsplit(img_name_landsat, '_L_');
    current_group = parts{1}; 

    fprintf('==============================================\n');
    fprintf('>>> 开始处理 [%d/%d]: %s \n', file_idx, length(tif_files), current_group);
    
    % 构建完整路径
    path_modis_t1  = fullfile(base_dir, 'MODIS_01', img_name_modis); 
    path_modis_t2  = fullfile(base_dir, 'MODIS_02', img_name_modis);
    path_landsat_t1 = fullfile(base_dir, 'Landsat_01', img_name_landsat);

    % 读取与格式化
    [MODIS_t1, info_m1, ~] = load_and_format_tif(path_modis_t1, data_range);
    [MODIS_t2, info_m2, ~] = load_and_format_tif(path_modis_t2, data_range);
    [Landsat_t1, info_l1, c_first] = load_and_format_tif(path_landsat_t1, data_range);

    % Fit-FC 核心运行逻辑
    J1 = Landsat_t1;
    I_MS1 = MODIS_t1;
    I_MS2 = MODIS_t2;
    [Number_row, Number_col, Number_bands] = size(Landsat_t1);

    I_MS0 = zeros(size(I_MS1));
    Z1    = zeros(size(J1));
    RB    = zeros(size(I_MS1));

    fprintf('>>> 正在执行引导滤波降维和残差提取...\n');
    for i = 1:Number_bands
        [a, b, q] = guidedfilter_MS_low(I_MS1(:,:,i), I_MS2(:,:,i), W);
        I_MS0(:,:,i) = a .* I_MS1(:,:,i) + b;
        Z1(:,:,i) = a .* J1(:,:,i) + b; 
        RB(:,:,i) = I_MS2(:,:,i) - I_MS0(:,:,i); 
    end

    Z2_interpolated = Z1 + RB;

    w0 = 20;  N_S = 20;  A = (2*w0 + 1) / 2;
    Z0 = zeros(size(Z1)); Z  = zeros(size(Z1));

    fprintf('>>> 正在执行 STARFM 残差分配核心...\n');
    tic
    for i = 1:Number_bands
        Z(:,:,i) = STARFM_fast_2016_v2(Z0(:,:,i), Z0(:,:,i), Z2_interpolated(:,:,i), J1(:,:,B1), J1(:,:,B2), w0, N_S, A);
    end
    fprintf('>>> 核心计算完成，耗时: %.2f 秒。\n', toc);

    % ==================== 结果保存与格式恢复 ===================
    Z_out = Z * diff(data_range) + data_range(1);

    bit_depth = info_l1(1).BitDepth;
    if bit_depth <= 8
        Z_out = uint8(Z_out);
    elseif bit_depth <= 16
        Z_out = uint16(Z_out);
    else
        Z_out = single(Z_out);
    end

    if c_first
        Z_out = permute(Z_out, [3, 1, 2]);
    end

    save_dir = fullfile(result_dir, 'FitFC_Results_L2');
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % 加上 current_group 让每次循环不覆盖，例如 Group_01_Pred_Landsat_t2.tif
    out_filename = fullfile(save_dir, sprintf('%s.tif', base_name));
    fprintf('>>> 保存结果到: %s\n', out_filename);

    try
        sz_out = size(Z_out);
        
        % 判断如果是 32 位浮点型（single），使用底层的 Tiff 类安全写入
        if isa(Z_out, 'single') || isa(Z_out, 'double')
            t = Tiff(out_filename, 'w');
            
            % 确定长、宽和波段数
            if length(sz_out) == 3 && c_first
                num_bands = sz_out(1); img_h = sz_out(2); img_w = sz_out(3);
            else
                img_h = sz_out(1); img_w = sz_out(2); num_bands = size(Z_out, 3);
            end

            % 配置 Tiff 头文件标签
            tagstruct.ImageLength = img_h;
            tagstruct.ImageWidth = img_w;
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 32; % 对应 single 浮点型
            tagstruct.SamplesPerPixel = 1;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            
            % 循环写入波段
            for b = 1:num_bands
                t.setTag(tagstruct);
                if length(sz_out) == 3 && c_first
                    t.write(single(squeeze(Z_out(b,:,:))));
                else
                    t.write(single(Z_out(:,:,b)));
                end
                
                if b ~= num_bands
                    t.writeDirectory(); % 为下一个波段开辟新页面
                end
            end
            t.close();
        else
            % 如果是正常的 uint8 / uint16，继续使用原来的 imwrite
            if length(sz_out) == 3 && c_first 
                imwrite(Z_out(1,:,:), out_filename);
                for b = 2:sz_out(1)
                    imwrite(Z_out(b,:,:), out_filename, 'WriteMode', 'append');
                end
            else
                imwrite(Z_out(:,:,1), out_filename);
                for b = 2:size(Z_out, 3)
                    imwrite(Z_out(:,:,b), out_filename, 'WriteMode', 'append');
                end
            end
        end
    catch ME
        warning('保存为 TIF 遇到问题 Error: %s', ME.message);
        save(strrep(out_filename, '.tif', '.mat'), 'Z_out');
    end
end
fprintf('\n>>> 所有影像处理完毕!\n');

end % <--- 这里结束了 run_fitfc 主函数


% ==================== 辅助方法 ===================
function [img_double, img_info, is_channels_first] = load_and_format_tif(file_path, data_range)
    % 尝试使用带地理信息的读取，方便后续完整写回
    info = imfinfo(file_path);
    num_images = numel(info);
    
    if num_images > 1 
        % 如果文件有多个 image 层（常见的带有C维度的 tif）
        tmp_img = imread(file_path, 1);
        [H, W] = size(tmp_img);
        C = num_images;
        img_raw = zeros(H, W, C, 'like', tmp_img);
        for img_idx = 1:num_images
            img_raw(:,:,img_idx) = imread(file_path, img_idx);
        end
        is_channels_first = false;
    else
        % 如果是直接读出了三维矩阵
        img_raw = imread(file_path);
    end
    
    img_info = info;

    % 处理 C=6 位置不确定的问题
    sz = size(img_raw);
    C_target = 6;
    is_channels_first = false;
    
    if length(sz) == 3
        if sz(1) == C_target
            % C, H, W -> H, W, C
            img_raw = permute(img_raw, [2, 3, 1]);
            is_channels_first = true;
        elseif sz(2) == C_target
            % H, C, W -> 此种情况极少，如果出现也做排列
            img_raw = permute(img_raw, [1, 3, 2]);
        end
    end
    
    % 将值域按给定范围标准化到 0~1 的 double 区间
    img_double = double(img_raw);
    if diff(data_range) ~= 0
        img_double = (img_double - data_range(1)) / diff(data_range);
    end
    % 边界截断，防止异常极值干扰计算
    img_double(img_double < 0) = 0;
    img_double(img_double > 1) = 1;
end
