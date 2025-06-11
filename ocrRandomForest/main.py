import numpy as np
from sklearn.ensemble import RandomForestClassifier
import struct
import gzip

def read_labels(path):
    with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.uint8)

def read_images(path):
    with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8)
        return data.reshape(num, rows * cols), rows, cols

def main():
    test_labels = read_labels("data/t10k-labels.idx1-ubyte")
    test_images, rows, cols = read_images("data/t10k-images.idx3-ubyte")
    print(f"Labels: {len(test_labels)}, Images: {test_images.shape[0]}, Rows: {rows}, Columns: {cols}")

    train_labels = read_labels("data/train-labels.idx1-ubyte")
    train_images, _, _ = read_images("data/train-images.idx3-ubyte")
    print(f"Labels: {len(train_labels)}, Images: {train_images.shape[0]}")

    clf = RandomForestClassifier()
    clf.fit(train_images, train_labels)

    score = clf.score(test_images, test_labels)
    print(f"Prediction score for Random Forest classifier {score*100:.2f}%")

    sample = 8
    pred = clf.predict(test_images[sample].reshape(1, -1))
    proba = clf.predict_proba(test_images[sample].reshape(1, -1))
    print(f"Sample {sample} prediction: {pred[0]}")
    print(f" Probabilities: {proba[0]}")
    print(f"Correct label: {test_labels[sample]}")

if __name__ == "__main__":
    main()
