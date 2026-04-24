/**
 * @file go2_vision_detection.cpp
 * @brief 基于 ONNX 模型的 Go2 机器人实时安全标识检测与动作响应主程序
 *
 * @par 使用说明
 *       go2_vision_detection <onnx_model_path> [classes_file] [network_interface]
 *       示例: ./go2_vision_detection safety_signs.onnx classes.txt eth0
 *       功能: 实时获取 Go2 机器人相机视频流，通过 ONNX 模型检测安全图标，
 *             根据检测结果自动触发机器人动作（伸懒腰、打招呼、闪灯）。
 *       控制: [q/Esc] 退出程序
 */

#include <unitree/robot/go2/video/video_client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

#include "LineProcessor.hpp"
#include "ONNXDetector.hpp"

/**
 * @brief Go2 机器人视觉检测控制器，集成视频流获取、ONNX 推理和动作控制
 */
class Go2VisionController {
public:
    /**
     * @brief 构造函数，初始化机器人客户端并加载 ONNX 模型
     * @param onnxModelPath ONNX 模型文件路径
     * @param classesPath 类别名称文本文件路径（可选）
     * @param netInterface 网络接口名称，如 eth0、wlan0（可选）
     */
    Go2VisionController(const std::string& onnxModelPath,
                        const std::string& classesPath = "",
                        const std::string& netInterface = "")
        : isRunning_(false)
        , lastDetectionTime_(std::chrono::steady_clock::now())
        , detectionCooldown_(2000) // 动作触发冷却时间 2 秒，防止重复触发
    {
        // 初始化网络接口
        if (!netInterface.empty()) {
            unitree::robot::ChannelFactory::Instance()->Init(0, netInterface);
        } else {
            unitree::robot::ChannelFactory::Instance()->Init(0);
        }

        // 初始化视频客户端
        videoClient_.SetTimeout(1.0f);
        videoClient_.Init();

        // 初始化运动控制客户端
        sportClient_.SetTimeout(10.0f);
        sportClient_.Init();

        // 初始化 VUI 客户端（用于前灯控制）
        vuiClient_.SetTimeout(1.0f);
        vuiClient_.Init();

        // 加载 ONNX 模型及可选类别文件
        if (!detector_.loadModel(onnxModelPath, classesPath)) {
            throw std::runtime_error("加载 ONNX 模型失败: " + onnxModelPath);
        }

        // 设置检测阈值
        detector_.setConfidenceThreshold(0.6f);
        detector_.setNMSThreshold(0.4f);

        std::cout << "Go2 视觉控制器初始化成功" << std::endl;
        std::cout << "已加载类别: ";
        auto classes = detector_.getClassNames();
        for (const auto& className : classes) {
            std::cout << className << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief 主循环：持续获取视频帧并进行检测，直到收到退出信号
     */
    void run() {
        isRunning_ = true;
        cv::namedWindow("Go2 Vision Detection", cv::WINDOW_AUTOSIZE);
        std::cout << "视觉检测已启动。按 'q' 键退出。" << std::endl;

        std::vector<uint8_t> imageSample;
        LineProcessor lineProcessor;

        while (isRunning_) {
            int ret = videoClient_.GetImageSample(imageSample);

            if (ret == 0 && !imageSample.empty()) {
                cv::Mat rawData(imageSample);
                cv::Mat frame = cv::imdecode(rawData, cv::IMREAD_COLOR);

                if (!frame.empty()) {
                    // 使用 LineProcessor 进行图像增强预处理
                    cv::Mat processedFrame = lineProcessor.process(imageSample);

                    // 执行 ONNX 目标检测
                    auto detections = detector_.detect(frame);

                    // 在图像上绘制检测结果（边界框和标签）
                    detector_.drawDetections(frame, detections);

                    // 显示检测画面
                    cv::imshow("Go2 Vision Detection", frame);

                    // 根据检测结果触发机器人动作
                    handleDetections(detections);

                    // 检测退出按键
                    char key = (char)cv::waitKey(1);
                    if (key == 'q' || key == 27) {
                        stop();
                    }
                }
            } else {
                // 无帧时仍响应退出按键
                if ((char)cv::waitKey(1) == 'q') {
                    stop();
                }
            }
        }

        cv::destroyAllWindows();
    }

    /**
     * @brief 停止主循环
     */
    void stop() {
        isRunning_ = false;
    }

private:
    /**
     * @brief 处理检测结果，根据检测到的安全标识触发对应机器人动作
     * @param detections 检测结果列表
     */
    void handleDetections(const std::vector<DetectionResult>& detections) {
        if (detections.empty()) {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        auto timeSinceLastDetection = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastDetectionTime_).count();

        // 冷却检查：防止频繁重复触发动作
        if (timeSinceLastDetection < detectionCooldown_) {
            return;
        }

        // 找出置信度最高的检测结果
        std::string highestClass;
        float highestConfidence = 0.0f;

        for (const auto& detection : detections) {
            if (detection.confidence > highestConfidence) {
                highestConfidence = detection.confidence;
                highestClass = detection.className;
            }
        }

        // 根据检测到的类别触发对应动作
        if (highestConfidence > 0.6f) {
            std::cout << "检测到: " << highestClass << " 置信度: "
                      << (highestConfidence * 100) << "%" << std::endl;

            if (highestClass == "caution_shock") {
                performStretch();
                lastDetectionTime_ = now;
            } else if (highestClass == "caution_oxidizer") {
                performHello();
                lastDetectionTime_ = now;
            } else if (highestClass == "caution_radiation") {
                flashFrontLights();
                lastDetectionTime_ = now;
            }
        }
    }

    /**
     * @brief 执行伸懒腰动作（对应电击警示标识）
     */
    void performStretch() {
        std::cout << ">>> 执行伸懒腰动作" << std::endl;
        try {
            sportClient_.Stretch();
            std::cout << ">>> 伸懒腰动作完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "!!! 伸懒腰动作失败: " << e.what() << std::endl;
        }
    }

    /**
     * @brief 执行打招呼动作（对应氧化剂警示标识）
     */
    void performHello() {
        std::cout << ">>> 执行打招呼动作" << std::endl;
        try {
            sportClient_.Hello();
            std::cout << ">>> 打招呼动作完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "!!! 打招呼动作失败: " << e.what() << std::endl;
        }
    }

    /**
     * @brief 闪烁前灯三次（对应辐射警示标识）
     */
    void flashFrontLights() {
        std::cout << ">>> 开始闪烁前灯 3 次" << std::endl;

        try {
            // 循环闪烁前灯 3 次
            for (int i = 0; i < 3; ++i) {
                // 开灯（亮度等级 10，最大值）
                int ret = vuiClient_.SetBrightness(10);
                if (ret != 0) {
                    std::cerr << "!!! 开灯失败，错误码: " << ret << std::endl;
                } else {
                    std::cout << "   第 " << (i + 1) << " 次闪烁 - 灯亮" << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(300));

                // 关灯（亮度等级 0）
                ret = vuiClient_.SetBrightness(0);
                if (ret != 0) {
                    std::cerr << "!!! 关灯失败，错误码: " << ret << std::endl;
                } else {
                    std::cout << "   第 " << (i + 1) << " 次闪烁 - 灯灭" << std::endl;
                }

                // 最后一次闪烁后不再等待
                if (i < 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                }
            }

            std::cout << ">>> 前灯闪烁完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "!!! 前灯闪烁失败: " << e.what() << std::endl;
        }
    }

private:
    unitree::robot::go2::VideoClient videoClient_; ///< 视频流客户端
    unitree::robot::go2::SportClient sportClient_; ///< 运动控制客户端
    unitree::robot::go2::VuiClient vuiClient_;     ///< VUI 客户端（前灯控制）
    ONNXDetector detector_;                        ///< ONNX 模型推理器

    std::atomic<bool> isRunning_;                                    ///< 主循环运行标志
    std::chrono::steady_clock::time_point lastDetectionTime_;        ///< 上次检测触发时间
    const int detectionCooldown_;                                    ///< 动作触发冷却时间（毫秒）

    std::mutex actionMutex_; ///< 动作执行互斥锁
};

/**
 * @brief 主函数，解析命令行参数并启动视觉检测控制器
 * @param argc 参数个数
 * @param argv 参数数组
 * @return int 正常退出返回 0，错误返回 -1
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <onnx_model_path> [classes_file] [network_interface]" << std::endl;
        std::cout << "参数说明:" << std::endl;
        std::cout << "  1. onnx_model_path  : ONNX 模型文件路径（必需）" << std::endl;
        std::cout << "  2. classes_file     : 类别名称文本文件路径（可选）" << std::endl;
        std::cout << "  3. network_interface: 网络接口名称，如 eth0、wlan0（可选）" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " safety_signs.onnx classes.txt eth0" << std::endl;
        std::cout << "  " << argv[0] << " safety_signs.onnx classes.txt" << std::endl;
        std::cout << "  " << argv[0] << " safety_signs.onnx" << std::endl;
        return -1;
    }

    std::string onnxModelPath = argv[1];
    std::string classesPath = (argc > 2) ? argv[2] : "";
    std::string netInterface = (argc > 3) ? argv[3] : "";

    // 判断第三个参数是否为网络接口（而非类别文件）
    // 规则：不带扩展名且以 eth/wlan/enp 开头则视为网络接口
    if (argc == 3) {
        std::string arg2 = argv[2];
        if (arg2.find('.') == std::string::npos &&
            (arg2.find("eth") == 0 || arg2.find("wlan") == 0 || arg2.find("enp") == 0)) {
            classesPath = "";
            netInterface = arg2;
        }
    }

    try {
        Go2VisionController controller(onnxModelPath, classesPath, netInterface);
        controller.run();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}