import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess_image(image):
    """Pré-processa a imagem para melhorar OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    if np.mean(binary) < 127:
        binary = 255 - binary
    
    return binary

def extract_digits_grid(image, rows, cols, start_row=0, start_col=0, num_rows=None, num_cols=None):
    """Extrai dígitos individuais de uma grade ou parte dela"""
    height, width = image.shape
    digit_height = height // rows
    digit_width = width // cols
    
    if num_rows is None:
        num_rows = rows - start_row
    if num_cols is None:
        num_cols = cols - start_col
    
    end_row = min(start_row + num_rows, rows)
    end_col = min(start_col + num_cols, cols)
    
    digits = []
    positions = []
    
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            y1 = row * digit_height
            y2 = (row + 1) * digit_height
            x1 = col * digit_width
            x2 = (col + 1) * digit_width
            
            digit = image[y1:y2, x1:x2]
            digits.append(digit)
            positions.append((row, col))
    
    return digits, positions

def generate_grid_visualization(original_image, processed_image, total_rows, total_cols, 
                               start_row, start_col, num_rows, num_cols):
    """Gera visualização do grid da área selecionada para processamento"""
    
    if len(original_image.shape) == 3:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    height, width = original_image.shape[:2]
    digit_height = height // total_rows
    digit_width = width // total_cols
    
    end_row = start_row + num_rows
    end_col = start_col + num_cols
    
    y1 = start_row * digit_height
    y2 = end_row * digit_height
    x1 = start_col * digit_width
    x2 = end_col * digit_width
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(image_rgb)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f'Imagem Original\nÁrea selecionada: linhas {start_row}-{end_row-1}, colunas {start_col}-{end_col-1}', fontsize=12)
    axes[0, 0].axis('off')
    
    selected_area = image_rgb[y1:y2, x1:x2]
    axes[0, 1].imshow(selected_area)
    
    for i in range(num_rows + 1):
        y = i * digit_height
        axes[0, 1].axhline(y=y, color='red', linewidth=1, alpha=0.8)
    
    for j in range(num_cols + 1):
        x = j * digit_width
        axes[0, 1].axvline(x=x, color='red', linewidth=1, alpha=0.8)
    
    if num_rows <= 10 and num_cols <= 15:
        for i in range(num_rows):
            for j in range(num_cols):
                center_x = j * digit_width + digit_width // 2
                center_y = i * digit_height + digit_height // 2
                abs_row = start_row + i
                abs_col = start_col + j
                axes[0, 1].text(center_x, center_y, f'{abs_row},{abs_col}', 
                              color='yellow', fontsize=8, fontweight='bold',
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
    
    axes[0, 1].set_title(f'Área Selecionada com Grid\n{num_rows}x{num_cols} células', fontsize=12)
    axes[0, 1].set_xlabel('Pixels (largura)')
    axes[0, 1].set_ylabel('Pixels (altura)')
    
    processed_area = processed_image[y1:y2, x1:x2]
    axes[1, 0].imshow(processed_area, cmap='gray')
    
    for i in range(num_rows + 1):
        y = i * digit_height
        axes[1, 0].axhline(y=y, color='red', linewidth=1, alpha=0.8)
    
    for j in range(num_cols + 1):
        x = j * digit_width
        axes[1, 0].axvline(x=x, color='red', linewidth=1, alpha=0.8)
    
    axes[1, 0].set_title('Área Processada (Binarizada)', fontsize=12)
    axes[1, 0].set_xlabel('Pixels (largura)')
    axes[1, 0].set_ylabel('Pixels (altura)')
    
    axes[1, 1].text(0.1, 0.9, f'Configurações do Grid:', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f'• Imagem total: {width}x{height} pixels', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'• Grid total: {total_rows} linhas × {total_cols} colunas', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'• Tamanho da célula: {digit_width}×{digit_height} pixels', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'• Área selecionada: {num_rows}×{num_cols} células', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'• Posição inicial: linha {start_row}, coluna {start_col}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f'• Total de dígitos a processar: {num_rows * num_cols}', fontsize=12, transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('selected_area_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualização do grid salva em 'selected_area_grid.png'")
    
    return digit_width, digit_height

def enhance_digit(digit_image):
    """Melhora um dígito individual para OCR"""
    pil_image = Image.fromarray(digit_image)
    pil_resized = pil_image.resize((60, 60), Image.LANCZOS)
    enhancer = ImageEnhance.Contrast(pil_resized)
    pil_enhanced = enhancer.enhance(2.0)
    return pil_enhanced

def process_digits_ocr(digits, positions):
    """Processa os dígitos extraídos usando OCR"""
    tesseract_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    results = []
    
    print(f"\nProcessando {len(digits)} dígitos...")
    
    for i, (digit, (row, col)) in enumerate(zip(digits, positions)):
        enhanced_digit = enhance_digit(digit)
        
        ocr_result = pytesseract.image_to_string(enhanced_digit, config=tesseract_config)
        ocr_clean = ocr_result.strip()
        
        if not ocr_clean or not ocr_clean.isdigit():
            alt_config = '--psm 8 -c tessedit_char_whitelist=0123456789'
            ocr_result = pytesseract.image_to_string(enhanced_digit, config=alt_config)
            ocr_clean = ocr_result.strip()
        
        if not ocr_clean or not ocr_clean.isdigit():
            ocr_clean = '?'
        
        results.append((row, col, ocr_clean))
        
        if (i + 1) % 10 == 0:
            print(f"Processados {i + 1}/{len(digits)} dígitos...")
    
    return results

def save_results(results, process_small_part, start_row, start_col):
    """Salva os resultados do OCR em arquivo"""
    result_grid = {}
    for row, col, digit in results:
        if row not in result_grid:
            result_grid[row] = {}
        result_grid[row][col] = digit
    
    recognized = sum(1 for _, _, r in results if r != '?')
    print(f"Dígitos reconhecidos: {recognized}/{len(results)} ({recognized/len(results)*100:.1f}%)")
    
    print("\nResultados por linha:")
    for row in sorted(result_grid.keys()):
        row_digits = []
        for col in sorted(result_grid[row].keys()):
            row_digits.append(result_grid[row][col])
        print(f"Linha {row+1:2d}: {''.join(row_digits)}")
    
    output_filename = 'digits_ocr_results_small.txt' if process_small_part else 'digits_ocr_results.txt'
    with open(output_filename, 'w') as f:
        f.write(f"Resultados OCR da imagem digits.png\n")
        if process_small_part:
            f.write(f"Área processada: linhas {start_row}-{start_row+len(result_grid)-1}, colunas {start_col}-{start_col+max(len(result_grid[r]) for r in result_grid)-1}\n")
        f.write("=" * 50 + "\n\n")
        
        for row in sorted(result_grid.keys()):
            row_digits = []
            for col in sorted(result_grid[row].keys()):
                row_digits.append(result_grid[row][col])
            f.write(f"Linha {row+1:2d}: {''.join(row_digits)}\n")
        
        f.write(f"\nEstatísticas:\n")
        f.write(f"Total de dígitos processados: {len(results)}\n")
        f.write(f"Reconhecidos: {recognized} ({recognized/len(results)*100:.1f}%)\n")
        f.write(f"Não reconhecidos: {len(results) - recognized}\n")
    
    print(f"Resultados salvos em '{output_filename}'")

def create_sample_visualization(digits, positions, results):
    """Cria visualização de uma amostra dos dígitos processados"""
    num_samples = min(10, len(digits))
    if num_samples > 0:
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
            
        for i in range(num_samples):
            axes[0, i].imshow(digits[i], cmap='gray')
            axes[0, i].set_title(f'Original\nPos({positions[i][0]},{positions[i][1]})\n{results[i][2]}', fontsize=8)
            axes[0, i].axis('off')
            
            enhanced = enhance_digit(digits[i])
            axes[1, i].imshow(np.array(enhanced), cmap='gray')
            axes[1, i].set_title(f'Enhanced\n{results[i][2]}', fontsize=8)
            axes[1, i].axis('off')
        
        plt.suptitle(f'Amostra dos primeiros {num_samples} dígitos processados', fontsize=14)
        plt.tight_layout()
        plt.savefig('processed_digits_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Amostra dos dígitos processados salva em 'processed_digits_sample.png'")

def main():
    PROCESS_SMALL_PART = True
    
    if PROCESS_SMALL_PART:
        start_row = 10
        start_col = 25
        num_rows_to_process = 30
        num_cols_to_process = 50
        print(f"Modo TESTE: Processando apenas {num_rows_to_process}x{num_cols_to_process} dígitos")
    else:
        start_row = 0
        start_col = 0
        num_rows_to_process = None
        num_cols_to_process = None
        print("Modo COMPLETO: Processando todos os 50x100 dígitos")
    
    image_path = 'digits.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return
    
    print(f"Imagem carregada: {image.shape}")
    
    processed = preprocess_image(image)
    
    total_rows = 50
    total_cols = 100
    
    if num_rows_to_process is None:
        num_rows_to_process = total_rows
    if num_cols_to_process is None:
        num_cols_to_process = total_cols
    
    print(f"\nGerando visualização do grid para a área selecionada...")
    generate_grid_visualization(image, processed, total_rows, total_cols, 
                               start_row, start_col, num_rows_to_process, num_cols_to_process)
    
    digits, positions = extract_digits_grid(
        processed, total_rows, total_cols, 
        start_row, start_col, 
        num_rows_to_process, num_cols_to_process
    )
    
    print(f"Extraídos {len(digits)} dígitos para processamento")
    
    results = process_digits_ocr(digits, positions)
    
    print(f"\nOCR concluído! Processados {len(results)} dígitos.")
    
    save_results(results, PROCESS_SMALL_PART, start_row, start_col)
    create_sample_visualization(digits, positions, results)

if __name__ == "__main__":
    main()