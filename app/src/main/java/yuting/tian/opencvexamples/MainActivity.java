package yuting.tian.opencvexamples;

import android.app.Activity;
import android.app.ActivityManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.ActionBarActivity;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public static final int Gray = 1;
    public static final int Canny = 2;
    public static final int Hist = 9;
    public static final int InnerWindowSobel = 4;
    public static final int Sepia = 5;
    public static final int Zoom = 6;
    public static final int Pixelize = 7;
    public static final int Posterize = 8;
    public static final int HoughLines = 3;

    private final static int MAX_SUPPORT_PROCESS_MODE = Hist;

    private CameraBridgeViewBase mOcvCameraView;
    private Button mBtnProgress = null;
    private Button mBtnCapture = null;

    private Mat mRgba;
    private Mat mGray;
    private Mat mTmp;
    private Mat mRgbaT;
    private Mat mRgbaF;

    private Size mSize0;
    private Mat mIntermediateMat;
    private MatOfInt mChannels[];
    private MatOfInt mHistSize;
    private int mHistSizeNum = 25;
    private Mat mMat0;
    private float[] mBuff;
    private MatOfFloat mRanges;
    private Point mP1;
    private Point mP2;
    private Scalar mColorsRGB[];
    private Scalar mColorsHue[];
    private Scalar mWhilte;
    private Mat mSepiaKernel;
    private int mProcessMethod = 0;

    private ImageView previewImg = null;
    private Bitmap captureBitmap = null;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    mOcvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        initView();
        mOcvCameraView.setVisibility(View.VISIBLE);

        initEvent();
    }


    private void initEvent() {
        mOcvCameraView.setCvCameraViewListener(this);

        mBtnProgress.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mProcessMethod++;
                if (mProcessMethod > MAX_SUPPORT_PROCESS_MODE) mProcessMethod = 0;
                String toastMsg = getModeName();
                mBtnProgress.setText(toastMsg);
                Toast.makeText(getBaseContext(), toastMsg, Toast.LENGTH_SHORT).show();
            }
        });

        mBtnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureBitmap = Bitmap.createBitmap(1280, 960, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mRgba, captureBitmap);

                mOcvCameraView.disableView();
                mOcvCameraView.setVisibility(SurfaceView.GONE);

                previewImg.setImageBitmap(captureBitmap);
                previewImg.setVisibility(View.VISIBLE);
            }
        });
    }

    private void initView() {
        mOcvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mBtnProgress = (Button) findViewById(R.id.buttonProgress);
        mBtnCapture = (Button) findViewById(R.id.buttonCapture);
        previewImg = (ImageView) findViewById(R.id.captureImage);

    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOcvCameraView != null) {
            mOcvCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOcvCameraView != null) {
            mOcvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        mTmp = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);

        mIntermediateMat = new Mat();
        mSize0 = new Size();
        mChannels = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mMat0 = new Mat();
        mColorsRGB = new Scalar[]{new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255)};
        mColorsHue = new Scalar[]{
                new Scalar(255, 0, 0, 255), new Scalar(255, 60, 0, 255), new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255), new Scalar(20, 255, 0, 255), new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255), new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255), new Scalar(0, 0, 255, 255), new Scalar(64, 0, 255, 255), new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255), new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mTmp.release();
        mRgbaT.release();
        mRgbaF.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat rgbaInnerWindow;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        switch (mProcessMethod) {
            case Gray:
                Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case Canny:
                mRgba = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mTmp, 80, 100);
                Imgproc.cvtColor(mTmp, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case Hist:
                Mat hist = new Mat();
                int thikness = (int) (sizeRgba.width / (mHistSizeNum + 10) / 5);
                if (thikness > 5) thikness = 5;
                int offset = (int) ((sizeRgba.width - (5 * mHistSizeNum + 4 * 10) * thikness) / 2);
                for (int c = 0; c < 3; c++) {
                    Imgproc.calcHist(Arrays.asList(mRgba), mChannels[c], mMat0, hist, mHistSize, mRanges);
                    Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
                    hist.get(0, 0, mBuff);
                    for (int h = 0; h < mHistSizeNum; h++) {
                        mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness;
                        mP1.y = sizeRgba.height - 1;
                        mP2.y = mP1.y - 2 - (int) mBuff[h];
                        Imgproc.line(mRgba, mP1, mP2, mColorsRGB[c], thikness);
                    }
                }
                Imgproc.cvtColor(mRgba, mTmp, Imgproc.COLOR_RGB2HSV_FULL);
                // Value
                Imgproc.calcHist(Arrays.asList(mTmp), mChannels[2], mMat0, hist, mHistSize, mRanges);
                Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
                hist.get(0, 0, mBuff);
                for (int h = 0; h < mHistSizeNum; h++) {
                    mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness;
                    mP1.y = sizeRgba.height - 1;
                    mP2.y = mP1.y - 2 - (int) mBuff[h];
                    Imgproc.line(mRgba, mP1, mP2, mWhilte, thikness);
                }
                break;
            case InnerWindowSobel:
                Mat gray = inputFrame.gray();
                Mat grayInnerWindow = gray.submat(top, top + height, left, left + width);
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.Sobel(grayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
                Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
                Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
                grayInnerWindow.release();
                rgbaInnerWindow.release();
                break;
            case Sepia:
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Core.transform(rgbaInnerWindow, rgbaInnerWindow, mSepiaKernel);
                rgbaInnerWindow.release();
                break;
            case Zoom:
                Mat zoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                Mat mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
                Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size());
                Size wsize = mZoomWindow.size();
                Imgproc.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
                zoomCorner.release();
                mZoomWindow.release();
                break;
            case Pixelize:
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.resize(rgbaInnerWindow, mIntermediateMat, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
                Imgproc.resize(mIntermediateMat, rgbaInnerWindow, rgbaInnerWindow.size(), 0., 0., Imgproc.INTER_NEAREST);
                rgbaInnerWindow.release();
                break;
            case Posterize:
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.Canny(rgbaInnerWindow, mIntermediateMat, 80, 90);
                rgbaInnerWindow.setTo(new Scalar(0, 0, 0, 255), mIntermediateMat);
                Core.convertScaleAbs(rgbaInnerWindow, mIntermediateMat, 1. / 16, 0);
                Core.convertScaleAbs(mIntermediateMat, rgbaInnerWindow, 16, 0);
                rgbaInnerWindow.release();
                break;
            case HoughLines:
                Imgproc.Canny(mRgba, mIntermediateMat, 80, 100);
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
                Mat lines = new Mat();
                Imgproc.HoughLines(mIntermediateMat, lines, 1, Math.PI / 180, 150);

                if (lines.cols() > 0) {
                    double[] data;
                    double rho, theta;
                    Point pt1 = new Point();
                    Point pt2 = new Point();
                    double a, b;
                    double x0, y0;
                    for (int i = 0; i < lines.cols(); i++) {
                        data = lines.get(0, i);
                        rho = data[0];
                        theta = data[1];
                        a = Math.cos(theta);
                        b = Math.sin(theta);
                        x0 = a * rho;
                        y0 = b * rho;
                        pt1.x = Math.round(x0 + 1000 * (-b));
                        pt1.y = Math.round(y0 + 1000 * a);
                        pt2.x = Math.round(x0 - 1000 * (-b));
                        pt2.y = Math.round(y0 - 1000 * a);
                        Imgproc.line(mRgba, pt1, pt2, mColorsRGB[1], 3);
                    }
                }
                break;
            default:
                mRgba = inputFrame.rgba();
                break;
        }


        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
        Core.flip(mRgbaF, mRgba, 1 );
        return mRgba;

    }

    private String getModeName() {
        switch (mProcessMethod) {
            case Gray:
                return "灰度";
            case Canny:
                return "边缘检测";
            case Hist:
                return "直方图";
            case InnerWindowSobel:
                return "索贝尔";
            case Sepia:
                return "褐色";
            case Zoom:
                return "缩放";
            case Pixelize:
                return "像素";
            case Posterize:
                return "色调分离";
            case HoughLines:
                return "霍夫线";
            default:
                return "progress";
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        mProcessMethod++;
        if (mProcessMethod > MAX_SUPPORT_PROCESS_MODE) mProcessMethod = 0;
        String toastMsg = getModeName();
        item.setTitle(toastMsg);
        Toast.makeText(getBaseContext(), toastMsg, Toast.LENGTH_SHORT).show();
        return true;
    }
}
