import os
from flask import Flask, render_template, request, session, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from helpers import *

app = Flask(__name__)
app.secret_key = 'visao_computacional'


PATH_UNTIL_PROJECT = 'D:/Guilherme/Documentos/Projetos Python/TCC/'
UPLOAD_FOLDER = PATH_UNTIL_PROJECT + 'static/images/originals/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/relatorio_imagem', methods=['POST',])
def relatorio_imagem():
    file = request.files['file']
    upload_path = app.config['UPLOAD_FOLDER']

    if (request.method == 'POST'):
        if 'file' not in request.files:
            flash('Nenhuma parte do arquivo', 'alert alert-danger')
            return redirect(url_for('index'))
        if (file.filename == ''):
            flash('Nenhum arquivo selecionado', 'alert alert-danger')
            return redirect(url_for('index'))
        if (file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_path, filename))
            # return redirect(url_for('uploaded_file',
            #                             filename=filename))
        else:
            flash('Extensão do arquivo não suportada', 'alert alert-danger')
            return redirect(url_for('index'))


    caminho_e_nome_do_arquivo = UPLOAD_FOLDER + file.filename
    imagem = carrega_imagem(caminho_e_nome_do_arquivo)


    caminho_imagem_jpg = upload_path.replace('originals', 'jpg') + salva_imagem_como_jpg(file.filename)
    salva_imagem(caminho_imagem_jpg, imagem)
    arquivo_jpg = 'images/jpg/' + salva_imagem_como_jpg(file.filename)


    imagem_suavizada = suaviza_imagem_metodo_bilateral(imagem)
    bin = transforma_para_binario(imagem_suavizada)
    # bin = transforma_para_binario_metodo_adaptativo(imagem_suavizada)
    # limiar, bin = transforma_para_binario_metodo_nobuyuki_otsu(imagem_suavizada)
    # imagem_tratada = operacao_morfologica_de_fechamento(bin)
    imagem_tratada = operacao_morfologica_de_abertura(bin)
    caminho_imagem_binaria = upload_path.replace('originals', 'bin') + salva_imagem_como_jpg(file.filename)
    salva_imagem(caminho_imagem_binaria, imagem_tratada)
    arquivo_binario = 'images/bin/' + salva_imagem_como_jpg(file.filename)

    # as informações e cálculos de área deve vir antes do metodo enumera_os_objetos_de_interesse(),
    # pois ele altera as propriedades da imagem original
    info = informacoes_imagem(imagem, caminho_e_nome_do_arquivo)

    # objetos uniformes
    pontos = coordenadas_dos_objetos_de_interesse(imagem)
    pontos_maximos_e_minimos = pontos_maximos_e_minimos_dos_objetos_de_interesse(pontos)
    diferenca = diferenca_entre_os_pontos(pontos_maximos_e_minimos)
    diametros_uniformes = diametro_de_objetos_uniformes(diferenca)


    # objetos não uniformes
    primeiros_e_ultimos_pontos = primeiro_e_ultimo_ponto(pontos)
    distancias = diferenca_entre_primeiro_e_ultimo_ponto(primeiros_e_ultimos_pontos)
    diametros_nao_uniformes = diametro_de_objetos_nao_uniformes(distancias, diferenca)

    # print(diametros_nao_uniformes)

    objetos_contornados = contorna_os_objetos_de_interesse(imagem)
    objetos_contornados_e_numerados = enumera_os_objetos_de_interesse(objetos_contornados)
    caminho_imagem_com_contornos = upload_path.replace('originals', 'com_contornos') + salva_imagem_como_jpg(file.filename)
    salva_imagem(caminho_imagem_com_contornos, objetos_contornados_e_numerados)
    arquivo_com_contornos = 'images/com_contornos/' + salva_imagem_como_jpg(file.filename)


    return render_template('relatorio_imagem.html',
                           diferenca=diferenca,
                           diametros_uniformes=diametros_uniformes,
                           diametros_nao_uniformes=diametros_nao_uniformes,
                           informacoes=info,
                           nome_arquivo=file.filename,
                           arquivo_jpg=arquivo_jpg,
                           arquivo_binario=arquivo_binario,
                           arquivo_com_contornos=arquivo_com_contornos)


# @app.route('/images/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


app.run(debug=True)
