#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * @brief Detection result structure
 */
struct Detection {
    int class_id;           ///< Class ID
    std::string class_name; ///< Class name
    float confidence;       ///< Confidence score (0-1)
    cv::Rect bbox;          ///< Bounding box (x, y, width, height)

    Detection() : class_id(-1), confidence(0.0f) {}
    Detection(int id, const std::string& name, float conf, const cv::Rect& box)
        : class_id(id), class_name(name), confidence(conf), bbox(box) {}
};

/**
 * @brief YOLOv8 detector using OpenCV DNN backend
 *
 * This class loads a YOLOv8 ONNX model and performs object detection
 * on input images. Supports GPU acceleration via OpenCV DNN CUDA backend.
 */
class YOLODetector {
public:
    /**
     * @brief Constructor
     * @param model_path Path to ONNX model file
     * @param class_names Vector of class names
     * @param input_size Input image size (default: 640)
     */
    YOLODetector(const std::string& model_path,
                 const std::vector<std::string>& class_names,
                 const cv::Size& input_size = cv::Size(640, 640));

    /**
     * @brief Destructor
     */
    ~YOLODetector();

    /**
     * @brief Initialize the detector
     * @param use_gpu Whether to use GPU acceleration (if available)
     * @return true if initialization succeeded, false otherwise
     */
    bool initialize(bool use_gpu = true);

    /**
     * @brief Detect objects in a single frame
     * @param frame Input image (BGR format)
     * @param confidence_threshold Minimum confidence threshold (0-1)
     * @param nms_threshold Non-Maximum Suppression threshold (0-1)
     * @return Vector of detected objects
     */
    std::vector<Detection> detect(const cv::Mat& frame,
                                 float confidence_threshold = 0.5f,
                                 float nms_threshold = 0.5f);

    /**
     * @brief Draw detection results on image
     * @param frame Image to draw on (modified in-place)
     * @param detections Detection results to draw
     * @param draw_confidence Whether to draw confidence scores
     */
    static void drawDetections(cv::Mat& frame,
                              const std::vector<Detection>& detections,
                              bool draw_confidence = true);

    /**
     * @brief Get the input size required by the model
     * @return Input image size
     */
    cv::Size getInputSize() const { return input_size_; }

    /**
     * @brief Check if detector is initialized
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const { return initialized_; }

private:
    // Disable copy constructor and assignment operator
    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;

    /**
     * @brief Preprocess input image for YOLOv8 model
     * @param frame Input image (BGR format)
     * @return Preprocessed blob
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * @brief Postprocess YOLOv8 output to detection results
     * @param output Model output tensor
     * @param frame_size Original frame size (for scaling boxes)
     * @param confidence_threshold Minimum confidence threshold
     * @param nms_threshold NMS threshold
     * @return Vector of filtered detections
     */
    std::vector<Detection> postprocess(const cv::Mat& output,
                                      const cv::Size& frame_size,
                                      float confidence_threshold,
                                      float nms_threshold);

    cv::dnn::Net net_;                      ///< OpenCV DNN network
    std::vector<std::string> class_names_;  ///< Class names
    cv::Size input_size_;                   ///< Input image size (width, height)
    bool initialized_;                      ///< Initialization flag
    std::vector<std::string> output_names_; ///< Output layer names
    bool use_gpu_;                          ///< GPU usage flag
};