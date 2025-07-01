import numpy as np
import cv2
import pytesseract
import matplotlib
matplotlib.use('Agg')


def sharpen_image(image):
    kernel = np.ones((3, 3), np.float32) / 90
    return cv2.filter2D(image, -1, kernel)


def apply_threshold(image, block_size, C):
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    ), block_size, C


def align_text(image, thresh_image):
    coords = np.column_stack(np.where(thresh_image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if abs(angle) < 2:
        angle = 0

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = 360

    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    return cv2.warpAffine(thresh_image, M, (w, h),
                         flags=cv2.INTER_CUBIC, 
                         borderMode=cv2.BORDER_REPLICATE)


def detect_text_rows(image):
    row_sums = np.sum(image == 255, axis=1)
    rows = []
    segment = []
    
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            segment.append(i)
        elif (row_sums[i] == 0) & (len(segment) >= 5):
            rows.append(segment)
            segment = []
        elif len(segment) > 0:
            rows.append(segment)
    
    return rows


def perform_ocr_on_rows(image, rows):
    custom_config = r'--oem 3 --psm 6 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    for i in range(len(rows)):
        try:
            row_img = image[rows[i][0]:rows[i][-1], :]
            
            if row_img.size == 0:
                print(f"Row {i}: Empty image segment, skipping...")
                continue
            
            if row_img.shape[0] < 5 or row_img.shape[1] < 5:
                print(f"Row {i}: Image segment too small ({row_img.shape}), skipping...")
                continue
                
            text = pytesseract.image_to_string(row_img, config=custom_config)
            print(f"Row {i}: '{text.strip()}'")
            
        except Exception as e:
            print(f"Row {i}: Error during OCR - {str(e)}")


def main():
    img = cv2.imread('image.png', 0)
    
    if img is None:
        print("Erro: Imagem não carregada. Verifique o caminho ou o nome do arquivo.")
        return
    
    img = sharpen_image(img)
    img_thresh, block_size, C = apply_threshold(img, 11, 2)
    img_thresh = 255 - img_thresh
    img_aligned = align_text(img, img_thresh)
    rows = detect_text_rows(img_aligned)
    perform_ocr_on_rows(img_aligned, rows)


if __name__ == "__main__":
    main()
