#define WIN32_LEAN_AND_MEAN

#include <WinSock2.h>
#include <WS2tcpip.h>
#include <windows.h>
#include <pylon/PylonIncludes.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <chrono>  // 시간 측정을 위해 추가
#include <filesystem>

#pragma comment(lib, "ws2_32.lib")

using namespace Pylon;
using namespace std;
using namespace GenApi;
using namespace cv;
using namespace std::chrono;  // 시간 측정을 위해 추가

static int cellCount = 0;

// 셀 분석 결과를 담는 구조체
struct CellAnalysis {
    bool hasObject;
    double angle;
    int cellNumber;
    Rect bbox;  // 셀의 위치 정보 추가
};

// ROI 추출 결과를 담는 구조체
struct ROIResult {
    Mat roi;
    Rect bbox;
    Mat binary;
    bool success;
    vector<CellAnalysis> cellResults;
};

// OpenCV 이미지 처리 클래스
class ImageProcessor {
public:
    static Mat pylonToMat(const CPylonImage& pylonImage) {
        if (pylonImage.IsValid()) {
            const uint8_t* pImageBuffer = (uint8_t*)pylonImage.GetBuffer();
            size_t width = pylonImage.GetWidth();
            size_t height = pylonImage.GetHeight();

            // 직접 Mat 생성 - clone() 없이 직접 데이터 복사
            Mat mat(height, width, CV_8UC3);
            memcpy(mat.data, pImageBuffer, width * height * 3);
            return mat;
        }
        return Mat();
    }

    // 셀 영역 자동 검출
    static vector<Rect> detectCellsAutomatically(const Mat& roi) {
        vector<Rect> cellRects;

        // 이진화를 위한 전처리
        Mat gray, thresh;
        cvtColor(roi, gray, COLOR_BGR2GRAY);
        threshold(gray, thresh, 150, 255, THRESH_BINARY_INV);  // 반전된 이진화

        imwrite("C:/GrabImages/binary_after_masking.png", thresh);

        // 노이즈 제거 및 셀 영역 강화
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

        // 컨투어 찾기
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 면적 기준으로 정렬
        sort(contours.begin(), contours.end(),
            [](const vector<Point>& c1, const vector<Point>& c2) {
                return contourArea(c1) > contourArea(c2);
            });

        // 면적이 가장 큰 첫 번째 컨투어를 제외하고 그 다음 20개 선택
        int maxCells = min(20, static_cast<int>(contours.size() - 2));  // 가장 큰 두 개를 제외
        for (int i = 2; i < maxCells + 2; i++) {  // i는 2부터 시작하되, maxCells + 2까지
            Rect bbox = boundingRect(contours[i]);
            cellRects.push_back(bbox);
        }

        // y 좌표로 정렬 후 각 행 내에서 x 좌표로 정렬
        sort(cellRects.begin(), cellRects.end(),
            [](const Rect& r1, const Rect& r2) {
                int row1 = r1.y / (r1.height + 10);  // 여유 간격 고려
                int row2 = r2.y / (r2.height + 10);
                if (row1 != row2) return row1 < row2;
                return r1.x < r2.x;
            });

        return cellRects;
    }

    // 셀 내 객체 존재 여부 확인
    static bool hasObject(const Mat& cell) {
        Mat gray, thresh;
        cvtColor(cell, gray, COLOR_BGR2GRAY);
        threshold(gray, thresh, 180, 255, THRESH_BINARY_INV);

        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 면적이 특정 임계값 이상인 컨투어가 있는지 확인
        double minArea = cell.rows * cell.cols * 0.01;  // 셀 면적의 1%
        for (const auto& contour : contours) {
            if (contourArea(contour) > minArea)
                return true;
        }
        return false;
    }

    static double measureAngle(const Mat& cell) {
        //imwrite("C:/GrabImages/cell_" + to_string(cellCount + 1) + "_original.png", cell);

        Mat hsv;
        cvtColor(cell, hsv, COLOR_BGR2HSV);

        // 각 색상별 HSV 범위 정의
        vector<pair<Scalar, Scalar>> colorRanges = {
            // 파란색
            {Scalar(80, 60, 70), Scalar(130, 255, 255)},
            // 검은색 (낮은 밝기값)
            {Scalar(0, 0, 0), Scalar(180, 255, 65)},
            // 노란색
            {Scalar(20, 100, 100), Scalar(30, 255, 255)},
            // 빨간색 (두 범위)
            {Scalar(0, 100, 100), Scalar(10, 255, 255)},
            {Scalar(170, 100, 100), Scalar(180, 255, 255)},
            // 보라색
            {Scalar(130, 50, 50), Scalar(150, 255, 255)},
            // 초록색
            {Scalar(35, 50, 50), Scalar(85, 255, 255)}
        };

        // 최종 마스크 초기화
        Mat finalMask = Mat::zeros(hsv.size(), CV_8UC1);
        vector<Mat> colorMasks;
        string pointColor = "";  // 점의 색상을 저장할 변수
        double cellArea = cell.rows * cell.cols;  // cellArea를 여기로 이동

        // 각 색상별 마스크 생성
        for (const auto& range : colorRanges) {
            Mat mask;
            inRange(hsv, range.first, range.second, mask);

            // 빨간색의 경우 두 번째 범위도 처리
            if (range.first == Scalar(0, 100, 100)) {
                Mat mask2;
                inRange(hsv, Scalar(170, 100, 100), Scalar(180, 255, 255), mask2);
                bitwise_or(mask, mask2, mask);
            }

            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);

            string colorName;
            if (range.first == Scalar(80, 60, 70)) colorName = "blue";
            else if (range.first == Scalar(0, 0, 0)) colorName = "black";
            else if (range.first == Scalar(20, 100, 100)) colorName = "yellow";
            else if (range.first == Scalar(0, 100, 100)) colorName = "red";
            else if (range.first == Scalar(130, 50, 50)) colorName = "purple";
            else if (range.first == Scalar(35, 50, 50)) colorName = "green";

            /*imwrite("C:/GrabImages/cell_" + to_string(cellCount + 1) + "_mask_" + colorName + ".png", mask);
            colorMasks.push_back(mask);*/

            // 점의 색상 확인 (두 번째로 큰 컨투어의 색상)
            vector<vector<Point>> contours;
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (!contours.empty()) {
                double maxArea = 0;
                for (const auto& contour : contours) {
                    double area = contourArea(contour);
                    if (area > maxArea && area < cellArea * 0.1) {  // 점은 전체 영역의 10% 미만
                        maxArea = area;
                        pointColor = colorName;
                    }
                }
            }

            bitwise_or(finalMask, mask, finalMask);
        }

        //imwrite("C:/GrabImages/cell_" + to_string(cellCount + 1) + "_mask_final.png", finalMask);

        int nonZeroCount = countNonZero(finalMask);
        bool hasObject = nonZeroCount > (cellArea * 0.002);

        Mat debug = cell.clone();
        double angle = 0.0;

        if (hasObject) {
            vector<vector<Point>> contours;
            findContours(finalMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            vector<vector<Point>> filteredContours;
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area > cellArea * 0.001) {
                    filteredContours.push_back(contour);
                }
            }

            if (filteredContours.size() > 1) {
                sort(filteredContours.begin(), filteredContours.end(),
                    [](const vector<Point>& c1, const vector<Point>& c2) {
                        return contourArea(c1) > contourArea(c2);
                    });

                RotatedRect lineRect = minAreaRect(filteredContours[0]);
                Point2f vertices[4];
                lineRect.points(vertices);

                // 선의 양 끝점 찾기
                Point2f lineStart, lineEnd;
                float maxDist = 0;
                for (int i = 0; i < 4; i++) {
                    float dist = norm(vertices[i] - vertices[(i + 1) % 4]);
                    if (dist > maxDist) {
                        maxDist = dist;
                        lineStart = vertices[i];
                        lineEnd = vertices[(i + 1) % 4];
                    }
                }

                // 선의 방향 벡터
                Point2f lineDir = lineEnd - lineStart;
                lineDir = lineDir / norm(lineDir);

                // 선 연장
                float extendLength = max(cell.cols, cell.rows) * 2.0f;
                Point2f extendedStart = lineStart - lineDir * extendLength;
                Point2f extendedEnd = lineEnd + lineDir * extendLength;

                // 점의 중심 계산
                Moments m = moments(filteredContours[1]);
                Point2f point(m.m10 / m.m00, m.m01 / m.m00);

                // 수선의 발 계산
                float a = lineDir.y;
                float b = -lineDir.x;
                float c = lineDir.x * lineStart.y - lineDir.y * lineStart.x;

                float t = (a * point.x + b * point.y + c) / (a * a + b * b);
                Point2f footPoint(point.x - a * t, point.y - b * t);

                Point2f perpVector = point - footPoint;

                angle = atan2(-perpVector.y, perpVector.x) * 180.0 / CV_PI;
                if (angle < 0) angle += 360.0;
                angle = fmod(450.0 - angle, 360.0);

                // 시각화
                line(debug, extendedStart, extendedEnd, Scalar(0, 255, 0), 2);
                line(debug, footPoint, point, Scalar(0, 0, 255), 2);
                circle(debug, point, 3, Scalar(0, 255, 255), -1);
                circle(debug, footPoint, 3, Scalar(255, 0, 0), -1);

                // 각도 표시
                int radius = 30;
                ellipse(debug, point, Size(radius, radius),
                    0, 0, angle,
                    Scalar(255, 255, 0), 2);

                // 텍스트 설정
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.5;
                int thickness = 1;
                string angleStr = to_string(int(round(angle)));

                // 텍스트 크기 계산
                Size angleSizeText = getTextSize(angleStr, fontFace, fontScale, thickness, nullptr);
                Size colorSizeText = getTextSize(pointColor, fontFace, fontScale, thickness, nullptr);

                // 텍스트 위치 계산 (오른쪽 위 모서리에서 10픽셀 떨어진 곳)
                Point anglePos(cell.cols - angleSizeText.width - 10, angleSizeText.height + 10);
                Point colorPos(cell.cols - colorSizeText.width - 10, angleSizeText.height + colorSizeText.height + 20);

                // 텍스트 배경을 위한 검은색 사각형 그리기
                rectangle(debug,
                    Point(cell.cols - max(angleSizeText.width, colorSizeText.width) - 15, 5),
                    Point(cell.cols - 5, angleSizeText.height + colorSizeText.height + 25),
                    Scalar(0, 0, 0), -1);

                // 텍스트 그리기
                putText(debug, angleStr, anglePos, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
                putText(debug, pointColor, colorPos, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
            }
        }
        else {
            // 텍스트 설정
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            string noObjectStr = "none object";

            // 텍스트 크기 계산
            Size textSize = getTextSize(noObjectStr, fontFace, fontScale, thickness, nullptr);

            // 텍스트 위치 계산 (오른쪽 위 모서리에서 10픽셀 떨어진 곳)
            Point textPos(cell.cols - textSize.width - 10, textSize.height + 10);

            // 텍스트 배경을 위한 검은색 사각형 그리기
            rectangle(debug,
                Point(cell.cols - textSize.width - 15, 5),
                Point(cell.cols - 5, textSize.height + 15),
                Scalar(0, 0, 0), -1);

            // 텍스트 그리기
            putText(debug, noObjectStr, textPos, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
        }

        imwrite("C:/GrabImages/cell_" + to_string(cellCount + 1) + "_result.png", debug);
        cellCount++;

        return angle;
    }

    // 흰색 판 ROI 추출
    static ROIResult extractWhitePlateROI(const Mat& image) {
        ROIResult result;
        result.success = false;

        // 타이밍 측정을 위한 변수 선언
        auto start_step = high_resolution_clock::now();
        auto end_step = high_resolution_clock::now();

        try {
            // 1. 그레이스케일 변환
            start_step = high_resolution_clock::now();
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            end_step = high_resolution_clock::now();
            cout << "1. Grayscale conversion: "
                << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

            // 2. 이진화
            start_step = high_resolution_clock::now();
            Mat thresh;
            threshold(gray, thresh, 170, 255, THRESH_BINARY);
            end_step = high_resolution_clock::now();
            cout << "2. Thresholding: "
                << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

            // 3. 모폴로지 연산
            start_step = high_resolution_clock::now();
            Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
            morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);
            end_step = high_resolution_clock::now();
            cout << "3. Morphological operations: "
                << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

            // 4. 컨투어 찾기
            start_step = high_resolution_clock::now();
            vector<vector<Point>> contours;
            findContours(thresh.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            end_step = high_resolution_clock::now();
            cout << "4. Finding contours: "
                << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

            if (!contours.empty()) {
                // 5. 최대 컨투어 찾기 및 ROI 추출
                start_step = high_resolution_clock::now();
                auto largest_contour = max_element(contours.begin(), contours.end(),
                    [](const vector<Point>& c1, const vector<Point>& c2) {
                        return contourArea(c1) < contourArea(c2);
                    });

                // 바운딩 박스 계산 및 여백 추가
                Rect bbox = boundingRect(*largest_contour);
                int padding = 10;
                bbox.x = max(0, bbox.x - padding);
                bbox.y = max(0, bbox.y - padding);
                bbox.width = min(image.cols - bbox.x, bbox.width + 2 * padding);
                bbox.height = min(image.rows - bbox.y, bbox.height + 2 * padding);

                // ROI 추출
                result.roi = image(bbox).clone();
                result.binary = thresh(bbox).clone();
                end_step = high_resolution_clock::now();
                cout << "5. ROI extraction: "
                    << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

                // 6. ROI 마스킹
                start_step = high_resolution_clock::now();
                int roiHeight = result.roi.rows;
                int maskStartY = roiHeight * 0.19;
                int maskEndY = roiHeight * 0.75;

                rectangle(result.roi,
                    Rect(0, maskStartY, result.roi.cols, maskEndY - maskStartY),
                    Scalar(0, 0, 0),
                    FILLED);

                result.bbox = bbox;
                result.success = true;
                end_step = high_resolution_clock::now();
                cout << "6. ROI masking: "
                    << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

                // 7. 셀 검출
                start_step = high_resolution_clock::now();
                vector<Rect> cellRects = detectCellsAutomatically(result.roi);
                end_step = high_resolution_clock::now();
                cout << "7. Cell detection: "
                    << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

                // 8. 각 셀 분석
                start_step = high_resolution_clock::now();
                for (int i = 0; i < cellRects.size(); i++) {
                    auto cell_start = high_resolution_clock::now();

                    Mat cellROI = result.roi(cellRects[i]);
                    CellAnalysis analysis;
                    analysis.cellNumber = i + 1;
                    analysis.bbox = cellRects[i];
                    analysis.hasObject = hasObject(cellROI);
                    analysis.angle = analysis.hasObject ? measureAngle(cellROI) : 0.0;
                    result.cellResults.push_back(analysis);

                    auto cell_end = high_resolution_clock::now();
                    cout << "    Cell " << (i + 1) << " analysis time: "
                        << duration_cast<microseconds>(cell_end - cell_start).count() << " us" << endl;
                }
                end_step = high_resolution_clock::now();
                cout << "8. Total cells analysis: "
                    << duration_cast<microseconds>(end_step - start_step).count() << " us" << endl;

                // 디버깅을 위한 셀 표시
                Mat debugImage = result.roi.clone();
                for (const auto& analysis : result.cellResults) {
                    rectangle(debugImage, analysis.bbox, Scalar(0, 255, 0), 2);
                    putText(debugImage, to_string(analysis.cellNumber),
                        Point(analysis.bbox.x, analysis.bbox.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                }
            }
        }
        catch (const exception& e) {
            cerr << "Error in ROI extraction: " << e.what() << endl;
        }
        return result;
    }
};

class TCPClient {
private:
    SOCKET sock;
    bool isConnected;

public:
    TCPClient() : sock(INVALID_SOCKET), isConnected(false) {
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw runtime_error("WSAStartup failed");
        }
    }

    ~TCPClient() {
        Disconnect();
        WSACleanup();
    }

    bool Connect(const string& ip, int port) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            return false;
        }

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &serverAddr.sin_addr);

        if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            closesocket(sock);
            return false;
        }

        isConnected = true;
        return true;
    }

    void Disconnect() {
        if (isConnected) {
            closesocket(sock);
            isConnected = false;
        }
    }

    bool SendMessage(const string& message) {
        if (!isConnected) return false;
        return send(sock, message.c_str(), static_cast<int>(message.length()), 0) != SOCKET_ERROR;
    }

    string ReceiveMessage() {
        if (!isConnected) return "";

        char buffer[1024];
        int bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0);

        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            cout << "Received from server: " << buffer << endl;
            return string(buffer);
        }
        else if (bytesReceived == 0) {
            isConnected = false;
            return "";
        }
        else if (bytesReceived == SOCKET_ERROR) {
            isConnected = false;
            return "";
        }
        return "";
    }
};

static high_resolution_clock::time_point actualGrabStartTime;

// 이벤트 핸들러 클래스 정의
class CGrabEventHandler : public CImageEventHandler
{
public:
    void OnImageGrabbed(CInstantCamera& camera, const CGrabResultPtr& grabResult) override
    {
        actualGrabStartTime = high_resolution_clock::now();
    }
};

// 카메라 관리 클래스
class BaslerCamera {
private:
    CInstantCamera camera;
    CGrabEventHandler* eventHandler;
    CImageFormatConverter formatConverter;
    CPylonImage pylonImage;

public:
    BaslerCamera() : eventHandler(nullptr) {
        try {
            PylonInitialize();
            formatConverter.OutputPixelFormat = PixelType_BGR8packed;
        }
        catch (const GenericException& e) {
            cerr << "Pylon 초기화 오류: " << e.GetDescription() << endl;
            throw;
        }
    }

    ~BaslerCamera() {
        Disconnect();
        PylonTerminate();
    }

    void Connect() {
        try {
            camera = CInstantCamera(CTlFactory::GetInstance().CreateFirstDevice());
            cout << "연결된 카메라: " << camera.GetDeviceInfo().GetModelName() << endl;
            camera.Open();
            
            eventHandler = new CGrabEventHandler();
            camera.RegisterImageEventHandler(eventHandler, RegistrationMode_Append, Cleanup_Delete);
            
            ConfigureCamera();
        }
        catch (const GenericException& e) {
            cerr << "카메라 연결 오류: " << e.GetDescription() << endl;
            throw;
        }
    }

    void Disconnect() {
        if (camera.IsOpen()) {
            camera.Close();
        }
    }

    Mat CaptureImage() {
        try {
            CGrabResultPtr ptrGrabResult;
            camera.GrabOne(INFINITE, ptrGrabResult);

            if (ptrGrabResult->GrabSucceeded()) {
                auto grabEndTime = high_resolution_clock::now();
                auto actualGrabDuration = duration_cast<microseconds>(grabEndTime - actualGrabStartTime);
                cout << "실제 이미지 획득 시간: " << actualGrabDuration.count() / 1000.0 << " ms" << endl;
                cout << "Line1 트리거로 이미지 캡처됨..." << endl;
                
                // 이미지 포맷 변환
                formatConverter.Convert(pylonImage, ptrGrabResult);
                Mat capturedImage = ImageProcessor::pylonToMat(pylonImage);
                
                pylonImage.Release();
                return capturedImage;
            }
            else {
                cerr << "이미지 캡처 실패" << endl;
                return Mat();
            }
        }
        catch (const GenericException& e) {
            cerr << "캡처 중 Pylon 오류: " << e.GetDescription() << endl;
            return Mat();
        }
        catch (const exception& e) {
            cerr << "캡처 중 표준 예외: " << e.what() << endl;
            return Mat();
        }
    }

private:
    void ConfigureCamera() {
        INodeMap& nodeMap = camera.GetNodeMap();

        // AcquisitionFrameCount 설정
        CIntegerPtr acquisitionFrameCount(nodeMap.GetNode("AcquisitionFrameCount"));
        if (IsAvailable(acquisitionFrameCount)) {
            acquisitionFrameCount->SetValue(1);
        }

        // TriggerSelector 설정
        CEnumerationPtr triggerSelector(nodeMap.GetNode("TriggerSelector"));
        if (IsAvailable(triggerSelector)) {
            triggerSelector->FromString("FrameStart");
        }

        // TriggerMode 설정
        CEnumerationPtr triggerMode(nodeMap.GetNode("TriggerMode"));
        if (IsAvailable(triggerMode)) {
            triggerMode->FromString("On");
        }

        // TriggerSource 설정
        CEnumerationPtr triggerSource(nodeMap.GetNode("TriggerSource"));
        if (IsAvailable(triggerSource)) {
            triggerSource->FromString("Line1");
        }

        // TriggerActivation 설정
        CEnumerationPtr triggerActivation(nodeMap.GetNode("TriggerActivation"));
        if (IsAvailable(triggerActivation)) {
            triggerActivation->FromString("FallingEdge");
        }

        // 노출 시간 설정
        CFloatPtr exposureTime(nodeMap.GetNode("ExposureTimeAbs"));
        if (IsAvailable(exposureTime)) {
            exposureTime->SetValue(47000.0);
        }
    }
};

// 셀 분석 전문 클래스
class CellAnalyzer {
private:
    const string saveDirectory;

public:
    CellAnalyzer(const string& directory = "C:/GrabImages/") : saveDirectory(directory) {
        // 저장 디렉토리 확인 및 생성
        if (!filesystem::exists(saveDirectory)) {
            filesystem::create_directories(saveDirectory);
        }
    }
    
    vector<CellAnalysis> AnalyzeROI(const ROIResult& result) {
        if (!result.success) {
            cerr << "ROI 추출 실패" << endl;
            return {};
        }
        
        return result.cellResults;
    }
    
    void SaveResults(const vector<CellAnalysis>& cellResults) {
        ofstream resultFile(saveDirectory + "analysis_results.txt");
        
        for (const auto& cellResult : cellResults) {
            resultFile << "Cell " << cellResult.cellNumber
                << ": Object Present = " << (cellResult.hasObject ? "Yes" : "No")
                << ", Angle = " << cellResult.angle << " degrees"
                << ", Position = [" << cellResult.bbox.x << ", "
                << cellResult.bbox.y << ", "
                << cellResult.bbox.width << ", "
                << cellResult.bbox.height << "]\n";

            cout << "Cell " << cellResult.cellNumber
                << ": Object Present = " << (cellResult.hasObject ? "Yes" : "No");

            if (cellResult.hasObject) {
                cout << ", Angle = " << cellResult.angle << " degrees";
            }
            cout << endl;
        }
        resultFile.close();
    }
    
    void SaveImage(const Mat& image, const string& filename) {
        imwrite(saveDirectory + filename, image);
    }
};

// 전체 프로세스 관리 클래스
class ProcessManager {
private:
    unique_ptr<BaslerCamera> camera;
    unique_ptr<TCPClient> tcpClient;
    unique_ptr<CellAnalyzer> cellAnalyzer;
    const string saveDirectory;

public:
    ProcessManager(const string& directory = "C:/GrabImages/") 
        : saveDirectory(directory) {
        camera = make_unique<BaslerCamera>();
        tcpClient = make_unique<TCPClient>();
        cellAnalyzer = make_unique<CellAnalyzer>(directory);
    }
    
    void Initialize() {
        // TCP 서버 연결
        if (!tcpClient->Connect("127.0.0.1", 8080)) {
            throw runtime_error("서버 연결 실패");
        }
        cout << "서버 연결 성공" << endl;
        
        // 카메라 초기화
        camera->Connect();
    }
    
    void ProcessLoop() {
        while (true) {
            cellCount = 0;  // 글로벌 변수 초기화
            
            // 전체 처리 시작 시간 측정
            auto totalStartTime = high_resolution_clock::now();
            
            // 이미지 캡처
            auto convertStartTime = high_resolution_clock::now();
            Mat capturedImage = camera->CaptureImage();
            
            if (capturedImage.empty()) {
                cerr << "캡처된 이미지를 로드할 수 없음" << endl;
                continue;
            }
            
            cellAnalyzer->SaveImage(capturedImage, "CapturedImage.png");
            
            auto convertEndTime = high_resolution_clock::now();
            auto convertDuration = duration_cast<milliseconds>(convertEndTime - convertStartTime);
            cout << "이미지 변환 시간: " << convertDuration.count() << " ms" << endl;
            
            // ROI 추출 및 분석
            auto roiStartTime = high_resolution_clock::now();
            ROIResult result = ImageProcessor::extractWhitePlateROI(capturedImage);
            auto roiEndTime = high_resolution_clock::now();
            auto roiDuration = duration_cast<milliseconds>(roiEndTime - roiStartTime);
            cout << "ROI 추출 시간: " << roiDuration.count() << " ms" << endl;
            
            if (result.success) {
                cout << "ROI 추출 성공" << endl;
                
                // 셀 분석 결과 처리
                auto cellsStartTime = high_resolution_clock::now();
                vector<CellAnalysis> cellResults = cellAnalyzer->AnalyzeROI(result);
                
                cout << "\n" << cellResults.size() << "개 셀 분석 중..." << endl;
                cellAnalyzer->SaveResults(cellResults);
                
                auto cellsEndTime = high_resolution_clock::now();
                auto cellsDuration = duration_cast<milliseconds>(cellsEndTime - cellsStartTime);
                cout << "\n전체 셀 처리 시간: " << cellsDuration.count() << " ms" << endl;
                
                // 전체 처리 완료 시간 측정
                auto totalEndTime = high_resolution_clock::now();
                auto totalDuration = duration_cast<microseconds>(totalEndTime - totalStartTime);
                cout << "\n전체 처리 시간: " << totalDuration.count() << " 마이크로초 ("
                    << fixed << setprecision(3) << totalDuration.count() / 1000.0 << " ms)" << endl;
                
                // 결과 전송
                string message = "이미지 처리 성공";
                if (tcpClient->SendMessage(message)) {
                    cout << "서버에 메시지 전송됨" << endl;
                }
                else {
                    cerr << "서버에 메시지 전송 실패" << endl;
                }
            }
            else {
                cerr << "ROI 추출 실패" << endl;
            }
        }
    }
    
    void Cleanup() {
        camera->Disconnect();
        tcpClient->Disconnect();
    }
};

int main()
{
    try {
        ProcessManager processManager;
        processManager.Initialize();
        processManager.ProcessLoop();
        processManager.Cleanup();
    }
    catch (const GenericException& e) {
        cerr << "Pylon 오류: " << e.GetDescription() << endl;
    }
    catch (const exception& e) {
        cerr << "표준 예외: " << e.what() << endl;
    }

    return 0;
}
