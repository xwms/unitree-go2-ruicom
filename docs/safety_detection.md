# Go2 安全标志检测与动作控制系统

## 项目概述

本项目为睿抗四足多模态赛道开发，实现了Go2机器狗的安全标志检测与智能动作控制功能。系统通过Go2摄像头获取实时视频流，使用YOLOv8深度学习模型检测安全标志，并根据检测结果控制机器狗执行相应动作。

### 核心功能
- 实时视频流获取与处理
- YOLOv8安全标志检测（3类：小心氧化剂、小心辐射、小心电击）
- 智能动作触发：检测到`caution_shock`时伸懒腰，检测到`caution_oxidizer`时打招呼，检测到`caution_radiation`时闪烁前灯三次
- 阈值控制：置信度阈值和连续帧检测防止误触发

## 设计思路

### 1. 架构设计
系统采用模块化设计，分离视频流处理、目标检测和动作控制三个核心模块：

```
视频流获取 (VideoClient) → 目标检测 (YOLODetector) → 动作控制 (SportClient)
         ↓                         ↓                         ↓
    实时图像采集              安全标志识别              动作执行决策
```

### 2. 模型转换策略
由于项目使用C++实现，需要将PyTorch训练的YOLOv8模型转换为C++可用的格式：

1. **格式选择**：选择ONNX作为中间格式，因其良好的兼容性和OpenCV DNN支持
2. **转换工具**：使用Ultralytics YOLO官方导出功能
3. **优化配置**：固定输入尺寸640x640，关闭动态尺寸，使用FP32精度

### 3. 检测逻辑设计
- **置信度阈值**：0.7，确保检测结果可靠
- **连续帧检测**：同一类别连续3帧检测到才触发动作，避免瞬时误判
- **冷却时间**：5秒内不重复触发同一动作

## 实现方案

### 1. 模型转换
```bash
yolo export model=best.pt format=onnx imgsz=640
```

### 2. C++核心类设计

#### YOLODetector类 (`include/YOLODetector.hpp`, `src/YOLODetector.cpp`)
```cpp
// 检测结果结构
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

// 主要接口
class YOLODetector {
public:
    YOLODetector(const std::string& model_path, 
                 const std::vector<std::string>& class_names);
    bool initialize(bool use_gpu = true);
    std::vector<Detection> detect(const cv::Mat& frame, 
                                  float confidence_threshold = 0.5f);
    static void drawDetections(cv::Mat& frame, 
                               const std::vector<Detection>& detections);
};
```

#### ActionTrigger类 (内嵌在 `src/go2_safety_detection.cpp`)
```cpp
class ActionTrigger {
public:
    bool shouldTrigger(const std::string& class_name, float confidence);
    void updateFrame(const std::vector<Detection>& detections);
private:
    std::map<std::string, std::deque<bool>> detection_history_;
    std::chrono::steady_clock::time_point last_action_time_;
    std::mutex mutex_;
};
```

### 3. 主程序流程 (`src/go2_safety_detection.cpp`)
```cpp
int main() {
    // 1. 初始化Unitree SDK
    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");
    
    // 2. 初始化视频和运动客户端
    unitree::robot::go2::VideoClient video_client;
    unitree::robot::go2::SportClient sport_client;
    
    // 3. 初始化YOLO检测器
    YOLODetector detector("best.onnx", class_names);
    
    // 4. 主循环
    while (true) {
        // 获取视频帧 → 运行检测 → 触发动作 → 显示结果
    }
}
```

## 文件结构

```
null_ruicom/
├── docs/
│   └── safety_detection.md          # 本文档
├── include/
│   ├── YOLODetector.hpp            # YOLO检测器头文件
│   └── LineProcessor.hpp
├── src/
│   ├── YOLODetector.cpp            # YOLO检测器实现
│   ├── go2_safety_detection.cpp    # 安全检测主程序
│   ├── test_yolo.cpp               # YOLO测试程序
│   ├── go2_video_client.cpp
│   ├── go2_sport_interactive.cpp
│   └── LineProcessor.cpp
├── dataset/
│   └── safety_signs_dataset/
│       └── runs/detect/train3/weights/best.pt  # 原始PyTorch模型
│       └── runs/detect/train3/weights/best.onnx  # onnx模型
└── CMakeLists.txt                  # CMake构建配置
```

## 使用教程

### 1. 环境准备
```bash
# 安装Python依赖
pip install torch onnx ultralytics

# 安装C++依赖 (Ubuntu/Debian)
sudo apt-get install libopencv-dev

# 使用conda安装OpenCV
conda install opencv
```

### 2. 模型转换
```bash
yolo export model=best.pt format=onnx imgsz=640
```

### 3. 编译项目
```bash
mkdir build && cd build
cmake .. -DOpenCV_DIR=/path/to/opencv/cmake -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 4. 运行程序
```bash
# 运行测试程序验证模型
./test_yolo

# 运行主程序（需要连接Go2机器狗）
./go2_safety_detection eth0
```

### 5. 程序控制
```
控制说明：
  [q] 或 [ESC] - 退出程序
  [s]          - 保存当前帧
  [其他]       - 继续运行

显示信息：
  Frame: XXX       - 已处理帧数
  Detections: X    - 当前帧检测到的对象数
  >>> Performing action for: class_name - 动作触发信息
```

## 测试方法

### 1. 单元测试
```bash
# 测试YOLO检测器
./test_yolo
```
测试程序会：
- 加载ONNX模型
- 读取测试图像
- 运行检测并显示结果
- 保存检测结果图像

### 2. 集成测试
1. **模型加载测试**：验证ONNX模型能否正确加载
2. **检测精度测试**：使用测试集验证检测准确率
3. **性能测试**：测量帧率（目标：≥10 FPS）
4. **动作触发测试**：验证阈值控制逻辑

### 3. 实时测试
```bash
# 连接Go2机器狗进行实时测试
./go2_safety_detection eth0
```

## 配置参数

### 检测参数 (`src/go2_safety_detection.cpp`)
```cpp
constexpr float CONFIDENCE_THRESHOLD = 0.7f;      // 置信度阈值
constexpr float NMS_THRESHOLD = 0.5f;             // 非极大值抑制阈值
constexpr int CONSECUTIVE_FRAMES_THRESHOLD = 3;   // 连续检测帧数
constexpr int ACTION_COOLDOWN_MS = 5000;          // 动作冷却时间(ms)
```

### 模型参数 (`scripts/convert_yolov8_to_onnx.py`)
```python
input_size = 640    # 输入图像尺寸
opset = 12          # ONNX算子集版本
dynamic = False     # 固定批次大小
half = False        # FP32精度
```

## 常见问题与解决方案

### 1. 检测性能低
**问题**：帧率低于预期
**解决**：
```cpp
// 调整检测参数
constexpr float CONFIDENCE_THRESHOLD = 0.8f;  // 提高阈值减少计算
constexpr int CONSECUTIVE_FRAMES_THRESHOLD = 2; // 减少连续检测帧数

// 使用GPU加速
detector.initialize(true);  // 启用GPU
```

### 2. 动作误触发
**问题**：机器狗动作触发过于频繁
**解决**：
```cpp
// 增加冷却时间
constexpr int ACTION_COOLDOWN_MS = 8000;

// 增加连续检测要求
constexpr int CONSECUTIVE_FRAMES_THRESHOLD = 5;
```

## 性能优化建议

### 1. GPU加速
```cpp
// 启用CUDA后端
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

### 2. 异步处理
```cpp
// 使用独立线程进行检测
std::thread detection_thread([&]() {
    auto detections = detector.detect(frame);
    // 发送结果到主线程
});
```

### 3. 模型优化
```bash
# 转换为TensorRT格式（NVIDIA GPU）
python3 -c "import torch; torch.onnx.export(...)"
# 使用TensorRT优化器
```

### 4. 图像预处理优化
```cpp
// 降低输入分辨率
cv::Size input_size(320, 320);  // 从640x640降低

// 使用ROI区域检测
cv::Rect roi(0, 0, frame.cols/2, frame.rows);
cv::Mat roi_frame = frame(roi);
```

## 扩展功能

### 1. 添加新动作
```cpp
// 在performAction函数中添加新动作
if (class_name == "caution_radiation") {
    std::cout << "    Action: Custom action" << std::endl;
    sport_client.CustomAction();
}
```

### 2. 多模型支持
```cpp
// 动态切换模型
class MultiModelDetector {
public:
    void loadModel(const std::string& model_path);
    void switchModel(const std::string& model_name);
};
```