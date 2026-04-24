/**
 * @file ONNXDetector.hpp
 * @brief ONNX 模型推理器的头文件，定义检测结果结构和推理器接口
 *
 * @par 功能定位
 *       提供 ONNXDetector 类的声明和 DetectionResult 结构体定义。
 *       作为视觉识别模块的推理抽象层，封装了模型加载、推理执行、
 *       结果解析与可视化的全部接口，支持 YOLO 格式输出的目标检测任务。
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief 检测结果结构体，包含类别名称、置信度和边界框
 */
struct DetectionResult {
    std::string className;   ///< 检测到的目标类别名称
    float confidence;        ///< 检测置信度（0.0 ~ 1.0）
    cv::Rect boundingBox;    ///< 目标边界框

    DetectionResult(const std::string& name, float conf, const cv::Rect& box)
        : className(name), confidence(conf), boundingBox(box) {}
};

/**
 * @brief ONNX 模型推理器，封装目标检测的完整流程
 */
class ONNXDetector {
public:
    /**
     * @brief 构造函数，初始化检测参数默认值
     */
    ONNXDetector();

    /**
     * @brief 析构函数
     */
    ~ONNXDetector();

    /**
     * @brief 加载 ONNX 模型和类别名称文件
     * @param modelPath ONNX 模型文件路径
     * @param classesPath 类别名称文本文件路径（可选）
     * @return bool 加载成功返回 true，失败返回 false
     */
    bool loadModel(const std::string& modelPath, const std::string& classesPath = "");

    /**
     * @brief 对输入图像执行目标检测
     * @param image 输入图像
     * @return std::vector<DetectionResult> 检测结果列表
     */
    std::vector<DetectionResult> detect(const cv::Mat& image);

    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief 设置 NMS 非极大值抑制阈值
     * @param threshold NMS 阈值
     */
    void setNMSThreshold(float threshold);

    /**
     * @brief 获取当前加载的类别名称列表
     * @return std::vector<std::string> 类别名称列表
     */
    std::vector<std::string> getClassNames() const;

    /**
     * @brief 在图像上绘制检测结果（边界框、标签和置信度）
     * @param image 要绘制的图像（会被修改）
     * @param detections 检测结果列表
     */
    void drawDetections(cv::Mat& image, const std::vector<DetectionResult>& detections);

private:
    cv::dnn::Net net_;                    ///< ONNX 网络模型
    std::vector<std::string> classNames_; ///< 类别名称列表

    float confidenceThreshold_; ///< 置信度阈值
    float nmsThreshold_;        ///< NMS 非极大值抑制阈值

    cv::Size inputSize_;   ///< 网络输入尺寸
    float scaleFactor_;    ///< 图像缩放因子
    cv::Scalar meanValues_; ///< 均值减除参数
    bool swapRB_;          ///< 是否交换 R 和 B 通道

    /**
     * @brief 获取网络输出层的名称列表
     * @return std::vector<std::string> 输出层名称列表
     */
    std::vector<std::string> getOutputsNames();

    /**
     * @brief 图像预处理：将输入帧转换为网络输入的 blob 格式
     * @param frame 原始输入图像
     * @param blob 输出的 blob 数据
     */
    void preprocess(const cv::Mat& frame, cv::Mat& blob);

    /**
     * @brief 从网络输出中提取边界框
     * @param output 网络输出的原始张量
     * @param frameSize 原始图像的尺寸
     * @return std::vector<cv::Rect> 边界框列表
     */
    std::vector<cv::Rect> getBoundingBoxes(const cv::Mat& output, const cv::Size& frameSize);

    /**
     * @brief 从网络输出中提取类别 ID 列表
     * @param output 网络输出的原始张量
     * @return std::vector<int> 类别 ID 列表
     */
    std::vector<int> getClassIds(const cv::Mat& output);

    /**
     * @brief 从网络输出中提取置信度列表
     * @param output 网络输出的原始张量
     * @return std::vector<float> 置信度列表
     */
    std::vector<float> getConfidences(const cv::Mat& output);

    /**
     * @brief 应用非极大值抑制（NMS），去除重叠的检测框
     * @param boxes 所有候选边界框
     * @param confidences 各边界框对应的置信度
     * @return std::vector<int> 保留的边界框索引
     */
    std::vector<int> applyNMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences);
};