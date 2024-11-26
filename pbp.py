from PIL import Image
import numpy as np
import tracemalloc

def mse(image1_path, image2_path):
    tracemalloc.start()
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    
    # Redimensionar para o mesmo tamanho
    img1 = img1.resize((min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1])))
    img2 = img2.resize(img1.size)
    
    # Converter para arrays NumPy e Normalizar
    array1 = np.array(img1) / 255.0
    array2 = np.array(img2) / 255.0

    # Calcular o quadrado da diferença por pixel
    diff = np.abs(array1 - array2)

    # Calcular o erro total como a soma de todas as diferenças
    total_error = np.sum(diff)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
    return total_error, diff

# Exemplo de uso
image1 = "img/circle2.png"
image2 = "img/square.png"
error, diff_image = mse(image1, image2)

print(f"Erro total entre as imagens: {error}")