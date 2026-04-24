/**
 * @file ONNXDetector.cpp
 * @brief ONNX 模型推理器的实现，提供目标检测（加载模型、预处理、推理、后处理、可视化）的完整流程
 *
 * @par 功能定位
 *       作为视觉识别模块的核心推理引擎，封装了 ONNX 模型的加载、图像预处理、
 *       YOLO 格式输出的解析、NMS 非极大值抑制以及检测结果的可视化绘制功能。
 *       支持自定义置信度阈值和 NMS 阈值，可配合 LineProcessor 等模块使用。
 */

#include "ONNXDetector.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

/**
 * @brief 构造函数，初始化检测参数默认值
 */
ONNXDetector::ONNXDetector()
    : confidenceThreshold_(0.5f)
    , nmsThreshold_(0.4f)
    , inputSize_(cv::Size(640, 640))
    , scaleFactor_(1.0 / 255.0)
    , meanValues_(cv::Scalar(0, 0, 0))
    , swapRB_(true)
{
}

/**
 * @brief 析构函数
 */
ONNXDetector::~ONNXDetector()
{
}

/**
 * @brief 加载 ONNX 模型和类别名称文件
 * @param modelPath ONNX 模型文件路径
 * @param classesPath 类别名称文本文件路径（每行一个类别名），为空则使用默认安全标识类别
 * @return bool 加载成功返回 true，失败返回 false
 */
bool ONNXDetector::loadModel(const std::string& modelPath, const std::string& classesPath)
{
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);

        if (net_.empty()) {
            std::cerr << "从 " << modelPath << " 加载 ONNX 模型失败" << std::endl;
            return false;
        }

        // 设置计算后端和推理目标设备
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // 加载类别名称文件（如果提供）
        if (!classesPath.empty()) {
            std::ifstream classFile(classesPath);
            if (classFile.is_open()) {
                std::string className;
                while (std::getline(classFile, className)) {
                    if (!className.empty()) {
                        classNames_.push_back(className);
                    }
                }
                classFile.close();
                std::cout << "已加载 " << classNames_.size() << " 个类别名称" << std::endl;
            } else {
                std::cerr << "警告: 无法打开类别名称文件: " << classesPath << std::endl;
            }
        }

        // 未加载到类别名称时，使用默认的安全标识类别
        if (classNames_.empty()) {
            classNames_ = {
                "caution_shock",
                "caution_oxidizer",
                "caution_radiation"
            };
            std::cout << "使用默认安全标识类别名称" << std::endl;
        }

        std::cout << "ONNX 模型加载成功: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "加载模型时 OpenCV 异常: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "加载模型时异常: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 设置置信度阈值
 * @param threshold 置信度阈值（自动限制在 [0.0, 1.0] 范围内）
 */
void ONNXDetector::setConfidenceThreshold(float threshold)
{
    confidenceThreshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

/**
 * @brief 设置 NMS 非极大值抑制阈值
 * @param threshold NMS 阈值（自动限制在 [0.0, 1.0] 范围内）
 */
void ONNXDetector::setNMSThreshold(float threshold)
{
    nmsThreshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

/**
 * @brief 获取当前加载的类别名称列表
 * @return std::vector<std::string> 类别名称列表
 */
std::vector<std::string> ONNXDetector::getClassNames() const
{
    return classNames_;
}

/**
 * @brief 图像预处理：将输入帧转换为网络输入的 blob 格式
 * @param frame 原始输入图像
 * @param blob 输出的 blob 数据
 */
void ONNXDetector::preprocess(const cv::Mat& frame, cv::Mat& blob)
{
    cv::dnn::blobFromImage(frame, blob, scaleFactor_, inputSize_, meanValues_, swapRB_, false);
}

/**
 * @brief 获取网络输出层的名称列表
 * @return std::vector<std::string> 输出层名称列表
 */
std::vector<std::string> ONNXDetector::getOutputsNames()
{
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net_.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net_.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

/**
 * @brief 从网络输出中提取边界框
 * @param output 网络输出的原始张量
 * @param frameSize 原始图像的尺寸
 * @return std::vector<cv::Rect> 边界框列表
 */
std::vector<cv::Rect> ONNXDetector::getBoundingBoxes(const cv::Mat& output, const cv::Size& frameSize)
{
    std::vector<cv::Rect> boxes;

    // 输出格式为 YOLO 格式: [batch, num_detections, 85]
    // 其中 85 = [中心x, 中心y, 宽, 高, 置信度, 类别概率...]

    const int numDetections = output.size[1];
    const int numClasses = output.size[2] - 5; // 5 个值为 x, y, w, h, confidence

    for (int i = 0; i < numDetections; ++i) {
        float confidence = output.at<float>(0, i, 4);

        if (confidence > confidenceThreshold_) {
            // 找出最大概率的类别
            int classId = 0;
            float maxClassProb = 0.0f;
            for (int j = 0; j < numClasses; ++j) {
                float classProb = output.at<float>(0, i, 5 + j);
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j;
                }
            }

            float finalConfidence = confidence * maxClassProb;
            if (finalConfidence > confidenceThreshold_) {
                // 获取归一化的边界框坐标（值域 0-1）
                float centerX = output.at<float>(0, i, 0);
                float centerY = output.at<float>(0, i, 1);
                float width = output.at<float>(0, i, 2);
                float height = output.at<float>(0, i, 3);

                // 转换为像素坐标
                int left = static_cast<int>((centerX - width / 2) * frameSize.width);
                int top = static_cast<int>((centerY - height / 2) * frameSize.height);
                int right = static_cast<int>((centerX + width / 2) * frameSize.width);
                int bottom = static_cast<int>((centerY + height / 2) * frameSize.height);

                // 确保坐标不超出图像边界
                left = std::max(0, left);
                top = std::max(0, top);
                right = std::min(frameSize.width - 1, right);
                bottom = std::min(frameSize.height - 1, bottom);

                boxes.push_back(cv::Rect(left, top, right - left, bottom - top));
            }
        }
    }

    return boxes;
}

/**
 * @brief 从网络输出中提取类别 ID 列表
 * @param output 网络输出的原始张量
 * @return std::vector<int> 类别 ID 列表
 */
std::vector<int> ONNXDetector::getClassIds(const cv::Mat& output)
{
    std::vector<int> classIds;

    const int numDetections = output.size[1];
    const int numClasses = output.size[2] - 5;

    for (int i = 0; i < numDetections; ++i) {
        float confidence = output.at<float>(0, i, 4);

        if (confidence > confidenceThreshold_) {
            int classId = 0;
            float maxClassProb = 0.0f;
            for (int j = 0; j < numClasses; ++j) {
                float classProb = output.at<float>(0, i, 5 + j);
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j;
                }
            }

            float finalConfidence = confidence * maxClassProb;
            if (finalConfidence > confidenceThreshold_) {
                classIds.push_back(classId);
            }
        }
    }

    return classIds;
}

/**
 * @brief 从网络输出中提取置信度列表
 * @param output 网络输出的原始张量
 * @return std::vector<float> 置信度列表
 */
std::vector<float> ONNXDetector::getConfidences(const cv::Mat& output)
{
    std::vector<float> confidences;

    const int numDetections = output.size[1];
    const int numClasses = output.size[2] - 5;

    for (int i = 0; i < numDetections; ++i) {
        float confidence = output.at<float>(0, i, 4);

        if (confidence > confidenceThreshold_) {
            float maxClassProb = 0.0f;
            for (int j = 0; j < numClasses; ++j) {
                float classProb = output.at<float>(0, i, 5 + j);
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                }
            }

            float finalConfidence = confidence * maxClassProb;
            if (finalConfidence > confidenceThreshold_) {
                confidences.push_back(finalConfidence);
            }
        }
    }

    return confidences;
}

/**
 * @brief 应用非极大值抑制（NMS），去除重叠的检测框
 * @param boxes 所有候选边界框
 * @param confidences 各边界框对应的置信度
 * @return std::vector<int> 保留的边界框索引
 */
std::vector<int> ONNXDetector::applyNMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences)
{
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nmsThreshold_, indices);
    return indices;
}

/**
 * @brief 对输入图像执行完整的目标检测流程
 * @param image 输入图像
 * @return std::vector<DetectionResult> 检测结果列表（包含类别、置信度、边界框）
 */
std::vector<DetectionResult> ONNXDetector::detect(const cv::Mat& image)
{
    std::vector<DetectionResult> detections;

    if (net_.empty() || image.empty()) {
        return detections;
    }

    try {
        // 图像预处理，生成网络输入 blob
        cv::Mat blob;
        preprocess(image, blob);

        // 设置网络输入
        net_.setInput(blob);

        // 执行前向推理
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, getOutputsNames());

        if (outputs.empty()) {
            return detections;
        }

        // 取第一个输出层的结果
        cv::Mat& output = outputs[0];

        // 分别提取边界框、类别 ID 和置信度
        std::vector<cv::Rect> boxes = getBoundingBoxes(output, image.size());
        std::vector<int> classIds = getClassIds(output);
        std::vector<float> confidences = getConfidences(output);

        // 应用 NMS 去除重叠框
        std::vector<int> indices = applyNMS(boxes, confidences);

        // 收集 NMS 后的最终检测结果
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            if (idx < classIds.size() && idx < confidences.size() && idx < boxes.size()) {
                int classId = classIds[idx];
                float confidence = confidences[idx];
                cv::Rect box = boxes[idx];

                if (classId < classNames_.size()) {
                    detections.emplace_back(classNames_[classId], confidence, box);
                }
            }
        }

    } catch (const cv::Exception& e) {
        std::cerr << "检测时 OpenCV 异常: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "检测时异常: " << e.what() << std::endl;
    }

    return detections;
}

/**
 * @brief 在图像上绘制检测结果（边界框、类别标签和置信度）
 * @param image 要绘制的图像（会被修改）
 * @param detections 检测结果列表
 */
void ONNXDetector::drawDetections(cv::Mat& image, const std::vector<DetectionResult>& detections)
{
    if (detections.empty()) {
        return;
    }

    // 各类别对应的颜色映射（BGR 格式）
    std::map<std::string, cv::Scalar> classColors = {
        {"caution_shock", cv::Scalar(0, 0, 255)},      // 红色
        {"caution_oxidizer", cv::Scalar(0, 255, 255)}, // 黄色
        {"caution_radiation", cv::Scalar(255, 0, 0)},  // 蓝色
        {"first_mark", cv::Scalar(0, 255, 0)},         // 绿色
        {"second_mark", cv::Scalar(255, 0, 255)},      // 品红
    };

    // 遍历每个检测结果并绘制
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        const std::string& className = detection.className;
        float confidence = detection.confidence;
        const cv::Rect& box = detection.boundingBox;

        // 获取该类别对应的颜色，未匹配时默认白色
        cv::Scalar color = classColors[className];
        if (color == cv::Scalar(0, 0, 0)) {
            color = cv::Scalar(255, 255, 255);
        }

        // 生成标签文本
        std::string label = className + ": " + std::to_string(static_cast<int>(confidence * 100)) + "%";

        // 绘制边界框
        cv::rectangle(image, box, color, 2);

        // 在框顶绘制标签背景（填充矩形）
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::rectangle(image,
                     cv::Point(box.x, box.y - labelSize.height - 5),
                     cv::Point(box.x + labelSize.width, box.y),
                     color, cv::FILLED);

        // 在框上绘制黑色标签文字
        cv::putText(image, label,
                   cv::Point(box.x, box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 同时在图像顶部绘制标签，方便远距离查看
        int yPos = 25 * (i + 1);
        cv::putText(image, label, cv::Point(10, yPos),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    // 当有检测结果时，绘制图像边框作为提示
    if (!detections.empty()) {
        cv::rectangle(image, cv::Point(0, 0),
                     cv::Point(image.cols - 1, image.rows - 1),
                     cv::Scalar(200, 200, 200), 1);
    }
}