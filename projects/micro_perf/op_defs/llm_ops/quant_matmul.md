# 基础参数配置

- dtype: 输入激活值的数据类型
- w_dtype: 权重值的数据类型
- compute_dtype: 实际矩阵计算的数据类型
- dst_dtype: 输出激活值的数据类型

# 可能支持的数据类型
- int8
    - 1 byte
    - 使用 torch.int8 表示一个数值
    
- int4
    - 0.5 byte
    - 使用 torch.int8 表示两个数值

- float8 (默认float8_e4m3fn)
    - 1 byte

- float8_e4m3fn
    - 1 byte

- float8_e5m2
    - 1 byte

- mxfp8_e4m3fn
    - 1 byte

- mxfp8_e5m2
    - 1 byte

- mxfp4_e2m1fn
    - 0.5 byte



# 拓展参数配置（Vendor可定义）
如果不是数据类型自身决定，或者其他参数指定，我们假定：
- 输入激活值采用 per_token 量化
- 权重值采用 per_channel 量化
- scale值采用 float32 表示
- 输入激活值和权重值都要转换成 compute_dtype 再进行计算。

Vendor可以根据自身需求和算法需求，设计不同的参数区分不同的实现方式，但必须提供默认值，包括但不限于：
- 量化方式和具体的配置，比如使用 per_group 量化，确定激活值和权重值的 group_size
- 量化 scale 的数据类型。
