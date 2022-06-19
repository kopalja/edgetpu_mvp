#include <tf_liter/edgetpu_backend/model.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    // 0) Load model
    tf_liter::edgetpu_backend::model model("../model/mnist_edgetpu.tflite");

    // 1) Load image
    cv::Mat image = cv::imread(argv[1]);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    // 2) Convert the images to a tensor.
    std::vector<std::uint8_t> model_input;
    for (int i = 0; i < gray.rows; ++i)
        model_input.insert(model_input.end(), gray.ptr<std::uint8_t>(i), gray.ptr<std::uint8_t>(i) + gray.cols);
    tf_liter::tensor image_input{model_input, {1, gray.rows, gray.cols}};
    tf_liter::tensor_dict_t feed_dict = {{"serving_default_input_1:0", image_input}};

    // 3) Infer model
    tf_liter::tensor_dict_t output_dict = model.run(feed_dict, {"StatefulPartitionedCall:0"});
    std::vector<std::uint8_t> results = output_dict.at("StatefulPartitionedCall:0").extract<std::uint8_t>();

    for (const auto &item : results) {
        std::cout << (int)item << std::endl;
    }

    return 0;
}