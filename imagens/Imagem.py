import numpy as np
import cv2 as cv
from matplotlib import pyplot as grafico


class Imagem:

    def __init__(self, caminho, nome):
        self._nome = nome
        self._caminho = caminho


    # Método getter que retorna a concatenação do caminho, nome e extensão da imagem para carregamento posterior
    @property
    def nome(self):
        return "{}/{}".format(self._caminho, self._nome)


    # Método privado da classe que faz e retorna o carregamento da imagem com a função da biblioteca OpenCV imread
    def carrega_imagem(self):
        imagem = cv.imread(self.nome)
        return imagem


    # Método que mostra a imagem através da função imshow da biblioteca OpenCV
    def mostra_imagem(self, imagem):
        cv.imshow("Imagem", imagem)
        cv.waitKey(0)  # espera pressionar qualquer tecla
        cv.destroyAllWindows()


    def salva_imagem(self, novo_nome_imagem, imagem_tratada):
        cv.imwrite("{}/{}".format(self._caminho, novo_nome_imagem), imagem_tratada)


    def transforma_para_escala_cinza(self):
        cinza = cv.cvtColor(self.carrega_imagem(), cv.COLOR_BGR2GRAY)
        return cinza

    def transforma_para_hsv(self):
        hsv = cv.cvtColor(self.carrega_imagem(), cv.COLOR_BGR2HSV)
        return hsv


    def identifica_canais_de_cores(self, imagem):
        if (len(imagem.shape) > 2):
            return True
        else:
            return False


    def informacoes_imagem(self, imagem):
        canais_de_cores = self.identifica_canais_de_cores(imagem)
        escala_cinza = "Imagens em escala de cinza não possuem canais de cores"

        return print("Número de linhas (eixo X): {}\n"
                     "Número de colunas (eixo Y): {}\n"
                     "Quantidade de canais de cores: {}\n"
                     "Total de pixels da imagem: {} px\n".format(imagem.shape[0],
                                                               imagem.shape[1],
                                                               imagem.shape[2] if (canais_de_cores) else escala_cinza,
                                                               imagem.size))



    def gera_histograma(self, imagem):
        if (len(imagem.shape) > 2):
            azul, verde, vermelho = cv.split(imagem)
            grafico.hist(azul.ravel(), 256, [0, 256])
            grafico.figure();
            grafico.hist(verde.ravel(), 256, [0, 256])
            grafico.figure();
            grafico.hist(vermelho.ravel(), 256, [0, 256])
            grafico.show()
        else:
            grafico.hist(imagem.ravel(), 256, [0, 256])
            grafico.show()



    def equaliza_imagem(self, imagem):
        canais_de_cores = self.identifica_canais_de_cores(imagem)
        if(canais_de_cores):
            matiz, saturacao, valor = cv.split(imagem)
            cv.equalizeHist(valor)
            hsv = cv.merge((matiz, saturacao, valor))
            equalizada = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            return equalizada
        else:
            equalizada = cv.equalizeHist(imagem)
            return equalizada


    def suaviza_imagem(self, imagem, tipo):

        # Tipo 1: FILTRO DE MÉDIA - filtro passa-baixas, suprimi o conteúdo de alta frequência, como as bordas mais nítidas da imagem.
        # Tipo 2: FILTRO GAUSSIANO - filtro passa-baixas, suprimi o conteúdo de alta frequência (não tanto quanto o FILTRO DE MÉDIA), como as bordas mais nítidas da imagem.
        # Tipo 3: FILTRO DE MEDIANA - filtro intermediário, preserva detalhes de alta frequência, como bordas ou contornos. Ideal para ruídos tipo "Sal e pimenta".
        # Tipo 4: FILTRO BILATERAL - filtro bilateral, indicado para suavizar a imagem preservando os detalhes de bordas e contornos.

        if(tipo == 1):
            filtro_media = cv.blur(imagem, (5,5))
            self.mostra_imagem(filtro_media)
            return filtro_media
        elif(tipo == 2):
            filtro_gaussiano = cv.GaussianBlur(imagem, (5,5), 0)
            self.mostra_imagem(filtro_gaussiano)
            return filtro_gaussiano
        elif(tipo == 3):
            filtro_de_mediana = cv.medianBlur(imagem, 5)
            self.mostra_imagem(filtro_de_mediana)
            return filtro_de_mediana
        elif(tipo == 4):
            filtro_bilateral = cv.bilateralFilter(imagem, 9, 75, 75)
            self.mostra_imagem(filtro_bilateral)
            return filtro_bilateral
        else:
            print("Tipo de filtro não disponível! Digite: 1 - FILTRO DE MÉDIA\n"
                  "2 - FILTRO GAUSSIANO\n"
                  "3 - FILTRO DE MEDIANA\n"
                  "4 - FILTRO BILATERAL")


    def _suaviza_imagem_para_binario(self):
        cinza = self.transforma_para_escala_cinza()
        filtro_bilateral = cv.bilateralFilter(cinza, 9, 75, 75)
        return filtro_bilateral


    def transforma_para_binario(self):
        suavizada = self._suaviza_imagem_para_binario()
        # ret, img_binarizada = cv.threshold(self.carrega_imagem(), 230, 255, cv.THRESH_BINARY)
        ret, img_binarizada = cv.threshold(suavizada, 140, 255, cv.THRESH_BINARY)
        return img_binarizada



    def detecta_bordas(self):
        cinza = self.transforma_para_escala_cinza()
        canny = cv.Canny(cinza, 120, 200)
        return canny
    #
    #
    # def encontra_contornos(self):
    #     images = self.carrega_imagem()
    #     img_binarizada = self.transforma_para_binario()
    #     contornos, hierarquia = cv.findContours(img_binarizada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     contornada = cv.drawContours(images, contornos, -1, (0, 255, 0), 3)
    #     return contornada



    def area_objetos_encontrados(self, img_binarizada, indice):
        modo = cv.RETR_TREE
        metodo = cv.CHAIN_APPROX_SIMPLE
        contornos, hierarquia = cv.findContours(img_binarizada, modo, metodo)
        objeto = contornos[indice]
        area_em_pixels = cv.contourArea(objeto)

        parametro = cv.contourArea(contornos[0])
        area_em_mm2 = (25*area_em_pixels)/parametro
        diametro = ((4*area_em_mm2)/3.14)**(1/2)

        return print("Objeto {}: Área em pixels: {} px - Área em mm²: {} mm² - Diâmetro: {} mm\n".format(indice, area_em_pixels, area_em_mm2, diametro))

    # def area_dos_objetos_segmentados(file):
    #     contornos, hierarquia = encontra_objetos_de_interesse(file)
    #     objetos = contornos
    #
    #     areas = []
    #     areas_em_pixels = []
    #     areas_em_mm2 = []
    #     diametros = []
    #     area_fisica_do_parametro = 25
    #
    #     for ind in range(0, len(objetos)):
    #         areas.append(cv.contourArea(objetos[ind]))
    #         parametro = min(areas)
    #     # parametro = areas[2]
    #
    #     for ind2 in range(0, len(objetos)):
    #         areas_em_pixels.append(cv.contourArea(objetos[ind2]))
    #
    #         resp1 = (area_fisica_do_parametro*areas_em_pixels[ind2])/parametro
    #         areas_em_mm2.append(round(resp1, 3))
    #
    #         resp2 = ((4*areas_em_mm2[ind2])/3.14)**(1/2)
    #         diametros.append(round(resp2, 3))
    #     return areas_em_pixels, areas_em_mm2, diametros




    def conta_pixels(self, imagem):
        totalPixelsPreto = 0
        totalPixelsBranco = 0
        for x in range(0, imagem.shape[0]):
            for y in range(0, imagem.shape[1]):
                if(imagem[x,y] == 255):
                    totalPixelsBranco += 1
                else:
                    totalPixelsPreto += 1

        total_pixels = imagem.size
        ferrita = round((totalPixelsBranco*100)/total_pixels)
        perlita = round((totalPixelsPreto*100)/total_pixels)

        return print('Total de pixels: {}px\n'
                     'Quantidade de pixels brancos: {}px\n'
                     'Quantidades de pixels preto: {}px\n'
                     'Porcentagem de Ferrita: {}%\n'
                     'Porcentagem de Perilta: {}%'.format(total_pixels, totalPixelsBranco, totalPixelsPreto, ferrita, perlita))



img = Imagem('C:/Users/guilherme.silva/PycharmProjects/Imagens/src/imagens/', 'analise-metalografica-aco-01.jpg')

imagem = img.carrega_imagem()

img.salva_imagem('teste.jpg', imagem)

