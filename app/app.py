import streamlit as st
import pandas as pd
import numpy as np
import time
from ultralytics import YOLO
from glob import glob
import os
import zipfile
from PIL import Image, ImageFile
import shutil
from datetime import datetime
import pymorphy2

ImageFile.LOAD_TRUNCATED_IMAGES = True
morph = pymorphy2.MorphAnalyzer()
model = {'base': YOLO(r'base_model.pt'), 'pro': YOLO(r'advanced_model.pt')}
if 'mchoice' not in st.session_state:
    st.session_state['mchoice'] = 'base'
statistic = {
    "Олень": 0,
    "Косуля": 0,
    "Кабарга": 0
}


def processing(files_list, mchoice):
    lst = []
    DIR = "processed"  # создать
    if not (os.path.isdir(DIR)):
        os.mkdir(DIR)
    to_cedik = []
    for file in files_list:
        lst.append(f"{DIR}\{file.name}")
        with open(f"{DIR}\{file.name}", "wb") as f:
            f.write(file.getbuffer())
    results = model[mchoice](lst)
    names = model[mchoice].names
    for i in range(len(results)):
        # results[i].save(filename=DIR + '\\'
        #                          + 'processed_' + lst[i].split('\\')[-1])  # save to disk
        try:
            result = names[int(results[i].boxes.cls[0])]
            for j in range(len(results[i].boxes.conf)):
                if results[i].boxes.conf[j] > 0.3:
                    to_cedik.append(names[int(results[i].boxes.cls[j])])
        except IndexError:
            result = 'Не_определено'
        if not (os.path.isdir(DIR + f"\\{result}")):
            os.mkdir(DIR + f"\\{result}")

        name = lst[i].split('\\')[-1]
        os.rename(lst[i], DIR + f"\{result}\{name}")

    files = glob(DIR + '\\*')
    with zipfile.ZipFile(f"{DIR}\\resulted.zip", "w") as zf:
        for filename in files:
            zf.write(filename)
            for f in glob(filename + '\\*'):
                zf.write(f)
            shutil.rmtree(filename, ignore_errors=True)
        zf.close()

    with open(f"{DIR}\\resulted.zip", "rb") as f:
        down = st.download_button(
            label="Загрузить",
            data=f,
            file_name=f"results_{datetime.now().strftime('%Y-%m-%dT%H:%M')}.zip",
            mime="application/zip"
        )
        if down:
            files.clear()
    os.remove(f"{DIR}\\resulted.zip")
    return to_cedik


st.set_page_config(
    page_title="Рога и копыта",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Рога и копыта v.0.1')
tab1, tab2, tab3 = st.tabs(["Обработка новых фото", "Статистика", "Дообучение"])

with tab1:
    st.subheader("Перенесите несколько фотографий")
    files = st.file_uploader("Перенесите ваши фото (.png, .jpeg, .jpg)",
                             accept_multiple_files=True, type=["png", "jpeg", "jpg"])
    button2 = st.button("Подтвердить", key="312")

    if button2:
        with st.spinner('Обработка'):
            res = processing(files, st.session_state['mchoice'])
            for animal in res:
                statistic[animal] += 1
        st.toast('Готово! Вы можете посмотреть количество животных каждого вида во вкладке "Статистика"', icon='👍')
        st.info('Чтобы очистить список фотографий нажмите F5', icon="ℹ️")

with tab2:
    st.subheader("Статистика последнего запроса")
    st.write(f"Обнаружено:")
    for i in statistic.keys():
        st.write(f"~ {statistic[i]} {morph.parse(i)[0].make_agree_with_number(statistic[i]).word}")

with tab3:
    st.subheader("Выберете вид животного для дообучения")
    choice = st.radio("Вид животных:", ("Олень", "Косуля", "Кабарга"))
    mchoice = st.session_state['mchoice']
    if choice:
        st.subheader("Перенесите несколько фотографий или архив")
        files2 = st.file_uploader("Перенесите ваши фото (.png, .jpeg, .jpg)",
                                  accept_multiple_files=True, type=["png", "jpeg", "jpg"], key="12345")
        button1 = st.button("Подтвердить")
        st.warning(
            r'При возникновении ошибки при дообучении, поменяйте'
            r' в файле "C:\Users\<Ваш Username>\AppData\Roaming\U'
            r'ltralytics\settings.yaml" настройку datasets_dir на "<Папка в которой запускаете программу>/datasets"',
            icon="⚠️")
        if button1:
            lst = []
            labels = []
            for file in files2:
                DIR = r"datasets\data\images\train"
                DIR1 = r"datasets\data\images\val"

                if not (os.path.isdir('datasets')):
                    os.mkdir('datasets')
                if not (os.path.isdir('datasets\data')):
                    os.mkdir('datasets\data')
                if not (os.path.isdir('datasets\data\images')):
                    os.mkdir('datasets\data\images')
                if not (os.path.isdir('datasets\data\labels')):
                    os.mkdir('datasets\data\labels')
                if not (os.path.isdir(r'datasets\data\images\train')):
                    os.mkdir(r'datasets\data\images\train')
                if not (os.path.isdir(r'datasets\data\labels\train')):
                    os.mkdir(r'datasets\data\labels\train')
                if not (os.path.isdir(r'datasets\data\images\val')):
                    os.mkdir(r'datasets\data\images\val')
                if not (os.path.isdir(r'datasets\data\labels\val')):
                    os.mkdir(r'datasets\data\labels\val')
                with open('data.yaml', 'w') as f:
                    f.write(
                        "path: ./data\ntrain: ./images/train\nval: ./images/val\nnames:\n  0: Кабарга\n  1: Косуля\n  2: Олень")

                lst.append(f"{DIR1}\{file.name}")
                with open(f"{DIR1}\{file.name}", "wb") as f:
                    f.write(file.getbuffer())
                lst.append(f"{DIR}\{file.name}")
                with open(f"{DIR}\{file.name}", "wb") as f:
                    f.write(file.getbuffer())
            with st.spinner('Обработка'):
                results = model[mchoice](lst)
                for r in range(len(results)):
                    if choice == "Олень":
                        coord = [2]
                    elif choice == "Косуля":
                        coord = [1]
                    else:
                        coord = [0]
                    coord = coord + results[r].boxes.xywhn[0].cpu().tolist()
                    print(coord)
                    nm = lst[r].split("\\")[-1]
                    with open(r"datasets\data\labels\val" + "\\" + nm.rstrip(nm.split(".")[-1]) + "txt", "w") as f:
                        f.write(" ".join(map(lambda x: str(x), coord)))
                    with open(r"datasets\data\labels\train" + "\\" + nm.rstrip(nm.split(".")[-1]) + "txt", "w") as f:
                        f.write(" ".join(map(lambda x: str(x), coord)))
                    labels.append(r"datasets\data\labels\train" + "\\" + nm.rstrip(nm.split(".")[-1]) + "txt")
                    labels.append(r"datasets\data\labels\val" + "\\" + nm.rstrip(nm.split(".")[-1]) + "txt")

                model[mchoice].train(data="data.yaml", epochs=1, batch=1)

                shutil.rmtree('datasets', ignore_errors=True)
            b = 'Базовая'
            bb = 'base'
            p = 'Продвинутая'
            st.toast(f'Готово! Модель "{b if mchoice == bb else p}" успешно дообучена!', icon='👍')

with st.sidebar:
    option = st.selectbox(
        "Выберите модель для распознавания",
        ("Базовая", "Продвинутая"))
    if option == "Продвинутая":
        st.warning("Продвинутая модель требует больше ресурсов", icon="⚠️")
        st.session_state['mchoice'] = 'pro'
    elif option == "Базовая":
        st.session_state['mchoice'] = 'base'

