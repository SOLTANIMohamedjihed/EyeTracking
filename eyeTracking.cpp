#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main() {
    cv::VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        std::cout << "Failed to open the video capture device." << std::endl;
        return -1;
    }

    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        std::cout << "Failed to load the face cascade classifier." << std::endl;
        return -1;
    }

    cv::CascadeClassifier eyeCascade;
    if (!eyeCascade.load("haarcascade_eye.xml")) {
        std::cout << "Failed to load the eye cascade classifier." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (videoCapture.read(frame)) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);

            cv::Mat faceROI = gray(face);
            std::vector<cv::Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

            for (const auto& eye : eyes) {
                cv::Point eyeCenter(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                cv::circle(frame, eyeCenter, 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("Eye Tracking Autofocus", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    videoCapture.release();

    return 0;
}