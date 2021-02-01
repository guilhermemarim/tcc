from builtins import print, map
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as grafico
from PIL import Image, ImageFilter


OBJETO_DE_INTERESSE_NA_COR_PRETA = cv.THRESH_BINARY
OBJETO_DE_INTERESSE_NA_COR_BRANCA = cv.THRESH_BINARY_INV
FONTE = cv.FONT_HERSHEY_SIMPLEX


RETANGULAR = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# array([
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1]], dtype=uint8)

ELIPITICO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50,50))
# array([
# [0, 0, 1, 0, 0],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [0, 0, 1, 0, 0]], dtype=uint8)

CRUZ = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
# array([
# [0, 0, 1, 0, 0],
# [0, 0, 1, 0, 0],
# [1, 1, 1, 1, 1],
# [0, 0, 1, 0, 0],
# [0, 0, 1, 0, 0]], dtype=uint8)


PERSONALIZADO = np.matrix([
[0, 0, 1, 0, 0],
[0, 1, 1, 1, 0],
[1, 1, 1, 1, 1],
[0, 1, 1, 1, 0],
[0, 0, 1, 0, 0]
], np.uint8)

def carrega_imagem(file):
    imagem = cv.imread(file)
    return imagem

def mostra_imagem(nome, file):
    cv.imshow(nome, file)
    cv.waitKey(0)  # espera pressionar qualquer tecla
    cv.destroyAllWindows()

def transforma_para_escala_cinza(file):
    cinza = cv.cvtColor(file, cv.COLOR_BGR2GRAY)
    return cinza

def transforma_para_hsv(file):
    hsv = cv.cvtColor(file, cv.COLOR_BGR2HSV)
    return hsv

def salva_imagem(nome_arquivo, imagem):
    # o nome_arquivo deve conter o path (caminho), o nome da imagem, e .sua_extensão (.jpg, .tif, .png)
    cv.imwrite(nome_arquivo, imagem)

def salva_imagem_como_jpg(nome_arquivo):
    # o nome_arquivo deve conter o path (caminho), o nome da imagem, e .sua_extensão (.jpg, .tif, .png)
    for ind in range(0, len(nome_arquivo)):
        if(nome_arquivo[ind] == '.'):
            indice_ponto = ind
    new_file = nome_arquivo[0:indice_ponto] + '.jpg'
    return new_file

def suaviza_imagem_metodo_filtro_de_media(file):
    filtro_media = cv.blur(file, (5,5))
    return filtro_media

def suaviza_imagem_metodo_filtro_gaussiano(file):
    filtro_gaussiano = cv.GaussianBlur(file, (5,5), 0)
    return filtro_gaussiano

def suaviza_imagem_metodo_filtro_de_mediana(file):
    filtro_mediana = cv.medianBlur(file, 5)
    return filtro_mediana

def suaviza_imagem_metodo_bilateral(file):
    filtro_bilateral = cv.bilateralFilter(file, 10, 150, 150)
    return filtro_bilateral

def transforma_para_binario(file):
    cinza = transforma_para_escala_cinza(file)
    # ret, bin = cv.threshold(cinza, 60, 255, OBJETO_DE_INTERESSE_NA_COR_BRANCA)
    ret, bin = cv.threshold(cinza, 95, 255, OBJETO_DE_INTERESSE_NA_COR_BRANCA)
    # ret ---> valor inicial do limiar
    return bin

def transforma_para_binario_metodo_adaptativo(file):
    metodo_media = cv.ADAPTIVE_THRESH_MEAN_C
    metodo_gaussiano = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    cinza = transforma_para_escala_cinza(file)
    bin = cv.adaptiveThreshold(cinza, 255, metodo_media, OBJETO_DE_INTERESSE_NA_COR_BRANCA, 11, 5)
    return bin

def transforma_para_binario_metodo_nobuyuki_otsu(file):
    tipo = OBJETO_DE_INTERESSE_NA_COR_BRANCA + cv.THRESH_OTSU
    cinza = transforma_para_escala_cinza(file)
    limiar, bin = cv.threshold(cinza, 0, 255, tipo)
    return limiar, bin

def operacao_morfologica_de_erosao(file):
    elemento_estruturante = ELIPITICO
    imagem_tratada = cv.erode(file, elemento_estruturante, iterations=2)
    return imagem_tratada

def operacao_morfologica_de_dilatacao(file):
    elemento_estruturante = ELIPITICO
    imagem_tratada = cv.dilate(file, elemento_estruturante, iterations=2)
    return imagem_tratada

def operacao_morfologica_de_abertura(file):
    elemento_estruturante = ELIPITICO
    imagem_tratada = cv.morphologyEx(file, cv.MORPH_OPEN, elemento_estruturante)
    return imagem_tratada

def operacao_morfologica_de_fechamento(file):
    elemento_estruturante = ELIPITICO
    imagem_tratada = cv.morphologyEx(file, cv.MORPH_CLOSE, elemento_estruturante)
    return imagem_tratada

def identifica_canais_de_cores(file):
    if (len(file.shape) > 2):
        return True
    else:
        return False

def equaliza_imagem(file):
    canais_de_cores = identifica_canais_de_cores(file)
    if(canais_de_cores):
        hsv1 = transforma_para_hsv(file)
        matiz, saturacao, valor = cv.split(hsv1)
        cv.equalizeHist(valor)
        hsv2 = cv.merge((matiz, saturacao, valor))
        imagem_equalizada = cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)
        return imagem_equalizada
    else:
        imagem_equalizada = cv.equalizeHist(file)
        return imagem_equalizada

def gera_histograma(file, nome_arquivo):
    # o nome_arquivo deve conter o path (caminho), o nome da imagem, e .sua_extensão (.jpg, .tif, .png)
    if (identifica_canais_de_cores(file)):
        azul, verde, vermelho = cv.split(file)
        grafico.hist(azul.ravel(), 256, [0, 256])
        # grafico.figure()
        grafico.hist(verde.ravel(), 256, [0, 256])
        # grafico.figure()
        grafico.hist(vermelho.ravel(), 256, [0, 256])
        # grafico.show()
    else:
        grafico.hist(file.ravel(), 256, [0, 256])
        grafico.show()

# grafico.hist(cinza.ravel(), 256, [0,256])
# grafico.xlabel('Quantidade de pixels')
# grafico.ylabel('Intensidade da luz')
# grafico.savefig('C:/Users/guilherme.silva/PycharmProjects/Visao_computacional/static/images/originals/grafico.jpg')
# grafico.close()

def encontra_objetos_de_interesse(file):
    imagem_suavizada = suaviza_imagem_metodo_bilateral(file)
    bin = transforma_para_binario(imagem_suavizada)
    # limiar, bin = transforma_para_binario_metodo_nobuyuki_otsu(imagem_suavizada)
    # imagem_tratada = operacao_morfologica_de_fechamento(bin)
    imagem_tratada = operacao_morfologica_de_abertura(bin)
    modo = cv.RETR_TREE
    metodo = cv.CHAIN_APPROX_SIMPLE
    contornos, hierarquia = cv.findContours(imagem_tratada, modo, metodo)
    return contornos, hierarquia

def contorna_os_objetos_de_interesse(file):
    contornos, hierarquia = encontra_objetos_de_interesse(file)
    objetos = contornos
    objetos_contornados = cv.drawContours(file, objetos, -1, (0, 255, 0), 10)
    return objetos_contornados

def enumera_os_objetos_de_interesse(file):
    contornos, hierarquia = encontra_objetos_de_interesse(file)
    objetos = contornos
    for ind in range(0, len(objetos)):
        x = objetos[ind][0][0][0]
        y = objetos[ind][0][0][1] - 40
        objetos_numerados = cv.putText(file, str(ind + 1), (x, y), FONTE, 5, (0, 255, 0), 9, cv.LINE_AA)
    return objetos_numerados

def informacoes_imagem(file, caminho_e_nome_do_arquivo):
    canais_de_cores = identifica_canais_de_cores(file)
    contornos, hierarquia = encontra_objetos_de_interesse(file)
    # print(im.format, im.size, im.mode)
    img = Image.open(caminho_e_nome_do_arquivo)
    info = []
    if(canais_de_cores):
        info.append(file.shape[0])              # quantidade de linhas
        info.append(file.shape[1])              # quantidade de colunas
        info.append(file.shape[2])              # quantidade de canais de cores
        info.append(file.size/3)                # quantidade de pixels ---> se possui mais de um canal de cor, deve se dividir o total de pixels pela quantidade de canais de cores
        info.append(img.format)                 # Formato da imagem (JPG, TIF, PNG)
        info.append(img.mode)                   # Padrão de cores (RGB, RGBA, GRAYSCALE, HSV)
        info.append(len(contornos))             # quantidade de objetos
        # info.append(img.info.get('dpi')[0])     # Resolução da imagem em dpi
    else:
        info.append(file.shape[0])              # quantidade de linhas
        info.append(file.shape[1])              # quantidade de colunas
        info.append(1)                          # quantidade de canais de cores
        info.append(file.size)                  # quantidade de pixels
        info.append(img.format)                 # Formato da imagem (JPG, TIF, PNG)
        info.append(img.mode)                   # Padrão de cores (RGB, RGBA, GRAYSCALE, HSV)
        info.append(len(contornos))             # quantidade de objetos
        # info.append(img.info.get('dpi')[0])     # Resolução da imagem em dpi
    return info


# nome_arquivo = 'C:/Users/guilherme.silva/PycharmProjects/Visao_computacional/imagens/amostra_13.jpg'
# img = carrega_imagem(nome_arquivo)
# suavizada_media = suaviza_imagem_metodo_filtro_de_media(img)
# suavizada_gaussiana = suaviza_imagem_metodo_filtro_gaussiano(img)
# suavizada_mediana = suaviza_imagem_metodo_filtro_de_mediana(img)
# suavizada_bilateral = suaviza_imagem_metodo_bilateral(img)
# bin = transforma_para_binario(suavizada_bilateral)
# morf = operacao_morfologica_de_abertura(bin)
# morf = operacao_morfologica_de_fechamento(bin)
# contornos, hierarquia = encontra_objetos_de_interesse(img)
# draw = contorna_os_objetos_de_interesse(img)
# numerados = enumera_os_objetos_de_interesse(draw)
# salva_imagem('C:/Users/guilherme.silva/PycharmProjects/Visao_computacional/imagens/amostra_13_teste.jpg', draw)



def coordenadas_dos_objetos_de_interesse(file):
    contornos, hierarquia = encontra_objetos_de_interesse(file)
    x = []
    y = []
    for ind in range(0, len(contornos)):
        objeto = contornos[ind]
        x.append(objeto[:, 0, 0])
        y.append(objeto[:, 0, 1])

    pontos = np.array([x, y])
                      #0, #1
    return pontos

# pontos = coordenadas_dos_objetos_de_interesse(img)


def pontos_maximos_e_minimos_dos_objetos_de_interesse(pontos):
    ponto_minimo_em_x = []
    ponto_maximo_em_x = []
    ponto_minimo_em_y = []
    ponto_maximo_em_y = []
    tamanho_do_vetor = len(pontos[0])

    for ind in range(0, tamanho_do_vetor):
        ponto_minimo_em_x.append(min(pontos[0,ind]))
        ponto_maximo_em_x.append(max(pontos[0,ind]))
        ponto_minimo_em_y.append(min(pontos[1,ind]))
        ponto_maximo_em_y.append(max(pontos[1,ind]))

    pontos_maximos_e_minimos = np.array([ponto_minimo_em_x,  #0
                                         ponto_maximo_em_x,  #1
                                         ponto_minimo_em_y,  #2
                                         ponto_maximo_em_y]) #3
    return pontos_maximos_e_minimos

# pontos_maximos_e_minimos = pontos_maximos_e_minimos_dos_objetos_de_interesse(pontos)


def diferenca_entre_os_pontos(pontos_maximos_e_minimos):
    diferenca_entre_os_pontos_eixo_x = []
    diferenca_entre_os_pontos_eixo_y = []
    tamanho_do_vetor = len(pontos_maximos_e_minimos[0])

    for ind in range(0, tamanho_do_vetor):

        diferenca_entre_os_pontos_eixo_x.append(pontos_maximos_e_minimos[1, ind] - pontos_maximos_e_minimos[0, ind])
        diferenca_entre_os_pontos_eixo_y.append(pontos_maximos_e_minimos[3, ind] - pontos_maximos_e_minimos[2, ind])
    diferenca = np.array([diferenca_entre_os_pontos_eixo_x,  #0
                          diferenca_entre_os_pontos_eixo_y]) #1
    return diferenca

# diferenca = diferenca_entre_os_pontos(pontos_maximos_e_minimos)


# print('')
# print('-----diferenca_entre_os_pontos-----')
# print(diferenca)
# print(diferenca[0,1])
# print(diferenca[1,1])


def diametro_de_objetos_uniformes(diferenca_entre_os_pontos):
    distancia_fisica_em_x = []
    distancia_fisica_em_y = []
    parametro_fisico = 20  # valor físico de comparação
    parametro_em_pixels_x = diferenca_entre_os_pontos[0, 1]
    parametro_em_pixels_y = diferenca_entre_os_pontos[1, 1]
    tamanho_do_vetor = len(diferenca_entre_os_pontos[0])

    for ind in range(0, tamanho_do_vetor):
        distancia_fisica_em_x.append(  round(  ((parametro_fisico * diferenca_entre_os_pontos[0, ind]) / parametro_em_pixels_x), 2  )  )
        distancia_fisica_em_y.append(  round(  ((parametro_fisico * diferenca_entre_os_pontos[1, ind]) / parametro_em_pixels_y), 2  )  )

    diametros_uniformes = np.array([distancia_fisica_em_x,  # 0
                                    distancia_fisica_em_y]) # 1
    return diametros_uniformes


# diametros_uniformes = diametro_de_objetos_uniformes(diferenca)
# print('')
# print('-----diametros_uniformes-----')
# print(diametros_uniformes)



def primeiro_e_ultimo_ponto(pontos):
    primeiro_ponto_em_x = []
    ultimo_ponto_em_x = []
    primeiro_ponto_em_y = []
    ultimo_ponto_em_y = []
    tamanho_do_vetor = len(pontos[0])

    for ind in range(0, tamanho_do_vetor):
        quadrante = (len(pontos[0, ind])-1) / 4
        primeiro_ponto_em_x.append(pontos[0, ind][math.ceil(3.5 * quadrante)])
        ultimo_ponto_em_x.append(pontos[0, ind][math.floor(0.45 * quadrante)])
        primeiro_ponto_em_y.append(pontos[1, ind][math.ceil(3.2 * quadrante)])
        ultimo_ponto_em_y.append(pontos[1, ind][math.floor(0.85 * quadrante)])
        # primeiro_ponto_em_x.append(pontos[0, ind][math.ceil(2.8 * quadrante)])
        # ultimo_ponto_em_x.append(pontos[0, ind][math.floor(1.12 * quadrante)])
        # primeiro_ponto_em_y.append(pontos[1, ind][math.ceil(3.2 * quadrante)])
        # ultimo_ponto_em_y.append(pontos[1, ind][math.floor(0.76 * quadrante)])
    primeiros_e_ultimos_pontos = np.array([primeiro_ponto_em_x, #0
                                        ultimo_ponto_em_x,      #1
                                        primeiro_ponto_em_y,    #2
                                        ultimo_ponto_em_y])     #3
    return primeiros_e_ultimos_pontos


# primeiros_e_ultimos_pontos = primeiro_e_ultimo_ponto(pontos)
# print('')
# print('-----primeiros_e_ultimos_pontos-----')
# print(primeiros_e_ultimos_pontos)

def diferenca_entre_primeiro_e_ultimo_ponto(primeiros_e_ultimos_pontos):

    tamanho_do_vetor = len(primeiros_e_ultimos_pontos[0])
    distancia_em_x = []
    distancia_em_y = []

    for ind in range(0, tamanho_do_vetor):
        # distancia_em_x.append( primeiros_e_ultimos_pontos[1, ind] - primeiros_e_ultimos_pontos[0, ind])
        # distancia_em_y.append(primeiros_e_ultimos_pontos[3, ind] - primeiros_e_ultimos_pontos[2, ind])
        distancia_em_x.append(   int(   math.sqrt(math.pow(   (   primeiros_e_ultimos_pontos[1, ind] - primeiros_e_ultimos_pontos[0, ind]   ), 2)   )   )   )
        distancia_em_y.append(   int(   math.sqrt(math.pow(   (   primeiros_e_ultimos_pontos[3, ind] - primeiros_e_ultimos_pontos[2, ind]   ), 2)   )   )   )


    distancias = np.array([distancia_em_x,  #0
                           distancia_em_y]) #1

    return distancias

# distancias = diferenca_entre_primeiro_e_ultimo_ponto(primeiros_e_ultimos_pontos)
# print('')
# print('-----distancias-----')
# print(distancias)


def diametro_de_objetos_nao_uniformes(distancias, diferencas):

    tamanho_do_vetor = len(distancias[0])
    parametro_fisico = 20
    parametro_em_pixels = diferencas[0, 1]
    diametros_nao_uniformes = []

    # for ind in range(0, 3):
    diametros_nao_uniformes.append('*')
    diametros_nao_uniformes.append('**' + str((parametro_fisico * parametro_em_pixels) / parametro_em_pixels))
    diametros_nao_uniformes.append((parametro_fisico * distancias[0, 2]) / parametro_em_pixels)

    return diametros_nao_uniformes

# diametros_nao_uniformes = diametro_de_objetos_nao_uniformes(distancias, diferenca)
# print('')
# print('-----diametros_nao_uniformes-----')
# print(diametros_nao_uniformes)



# for ind in range(0, len(contornos)):
#     objetos_enumerados = cv.putText(draw, '.', (primeiros_e_ultimos_pontos[0, 2], primeiros_e_ultimos_pontos[2, 2]), FONTE, 10,
#                                     (255, 0, 0), 10, cv.LINE_AA)
# salva_imagem('E:/Documentos/alura_cursos/PYTHON/visao_computacional/imagens/teste.jpg', objetos_enumerados)
#
# for ind3 in range(0, len(contornos)):
#     objetos_enumerados = cv.putText(draw, '.', (primeiros_e_ultimos_pontos[1, 2], primeiros_e_ultimos_pontos[3, 2]), FONTE, 10,
#                                     (0, 0, 255), 10, cv.LINE_AA)
# salva_imagem('E:/Documentos/alura_cursos/PYTHON/visao_computacional/imagens/teste2.jpg', objetos_enumerados)









































# primeiro_ponto_em_x = []
# ultimo_ponto_em_x = []
# primeiro_ponto_em_y = []
# ultimo_ponto_em_y = []
# distancia = []


# for ind2 in range(0, len(x)):
#
#     media_x = int(len(x[ind2])/2)
#     tamanho = len(x[ind2])/4
#
#     primeiro_ponto_em_x.append(x[ind2][math.ceil(1.35*tamanho)])
#     ultimo_ponto_em_x.append(x[ind2][math.floor(3*tamanho)])
#
#     ponto_minimo_em_x.append(min(x[ind2]))
#     ponto_maximo_em_x.append(max(x[ind2]))
#     ponto_medio_em_x.append(x[ind2][media_x])
#
#     primeiro_ponto_em_y.append(y[ind2][math.ceil(1.35*tamanho)])
#     ultimo_ponto_em_y.append(y[ind2][math.floor(3*tamanho)])
#
#     metade_y = int(len(y[ind2])/2)
#     media_y = math.ceil(metade_y/2)
#     ponto_minimo_em_y.append(min(y[ind2]))
#     ponto_maximo_em_y.append(max(y[ind2]))
#     ponto_medio_em_y.append(y[ind2][media_y])
#
#     diferenca_entre_os_pontos_eixo_x.append(ponto_maximo_em_x[ind2] - ponto_minimo_em_x[ind2])
#     diferenca_entre_os_pontos_eixo_y.append(ponto_maximo_em_y[ind2] - ponto_minimo_em_y[ind2])
#
#     # math.sqrt(math.pow((ultimo_ponto_em_x[ind2] - primeiro_ponto_em_x[ind2]), 2) + math.pow((ultimo_ponto_em_y[ind2] - primeiro_ponto_em_y[ind2]), 2))
#     # distancia.append(math.sqrt(math.pow((math.sqrt(math.pow((ultimo_ponto_em_x[ind2] - primeiro_ponto_em_x[ind2]),2) + math.pow((ultimo_ponto_em_y[ind2] - primeiro_ponto_em_y[ind2]),2))), 2)))
#     distancia.append(   math.sqrt(math.pow((        ultimo_ponto_em_x[ind2] - primeiro_ponto_em_x[ind2]       ), 2))       )
#
#
# for ind3 in range(0, len(contornos)):
#     objetos_enumerados = cv.putText(draw, '.', (primeiro_ponto_em_x[ind3], primeiro_ponto_em_y[ind3]), FONTE, 10,
#                                     (255, 0, 0), 10, cv.LINE_AA)
# salva_imagem('C:/Users/guilherme.silva/PycharmProjects/Visao_computacional/imagens/teste.jpg', objetos_enumerados)
#
# for ind3 in range(0, len(contornos)):
#     objetos_enumerados = cv.putText(draw, '.', (ultimo_ponto_em_x[ind3], ultimo_ponto_em_y[ind3]), FONTE, 10,
#                                     (0, 0, 255), 10, cv.LINE_AA)
# salva_imagem('C:/Users/guilherme.silva/PycharmProjects/Visao_computacional/imagens/teste2.jpg', objetos_enumerados)
#
#
#
# tamanho = len(distancia)
# diferenca_fisica = []
# distancia_fisica = []
# area_fisica = []
#
#
# print(diferenca_entre_os_pontos_eixo_x)
# print(distancia)
#
# for ind4 in range(0, tamanho):
#     diferenca_fisica.append((22*diferenca_entre_os_pontos_eixo_x[ind4])/diferenca_entre_os_pontos_eixo_x[0])
#     distancia_fisica.append((22*distancia[ind4])/diferenca_entre_os_pontos_eixo_x[0])
#
# print("---------------------")
# print(diferenca_fisica)
# print(distancia_fisica)




