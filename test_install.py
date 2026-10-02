import sys

print("=" * 50)
print("SIGN LANGUAGE PROJECT - INSTALLATION TEST")
print("=" * 50)

print("\nPython version:")
print(sys.version)

print("\nTesting NumPy...")
try:
    import numpy
    print("NumPy: OK")
except Exception as e:
    print("NumPy: ERROR")
    print(e)

print("\nTesting OpenCV...")
try:
    import cv2
    print("OpenCV: OK")
    print("OpenCV version:", cv2.__version__)
except Exception as e:
    print("OpenCV: ERROR")
    print(e)

print("\nTesting MediaPipe...")
try:
    import mediapipe
    print("MediaPipe: OK")
    print("MediaPipe version:", mediapipe.__version__)
except Exception as e:
    print("MediaPipe: ERROR")
    print(e)

print("\nTesting Joblib...")
try:
    import joblib
    print("Joblib: OK")
except Exception as e:
    print("Joblib: ERROR")
    print(e)

print("\nTesting Scikit-learn...")
try:
    import sklearn
    print("Scikit-learn: OK")
    print("Scikit-learn version:", sklearn.__version__)
except Exception as e:
    print("Scikit-learn: ERROR")
    print(e)

print("\n" + "=" * 50)
print("INSTALLATION TEST COMPLETED")
print("=" * 50)