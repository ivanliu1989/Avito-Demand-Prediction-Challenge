from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


def text_mining_v1(dat, n_comp=3):

    print('NLP - tfidf')
    # Get Russian Stopwords
    stopWords = stopwords.words('russian')

    # Create tfidf matrix for title and description
    # tfidf = TfidfVectorizer(max_features=50000, stop_words=stopWords)

    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        # strip_accents='unicode',
        analyzer='word',
        # token_pattern=r'\w{1,}',
        stop_words=stopWords,
        ngram_range=(1, 3),
        max_features=50000,
        norm='l2',
        min_df=3,
        max_df=0.6)

    tfidf_title = TfidfVectorizer(
        sublinear_tf=True,
        # strip_accents='unicode',
        analyzer='word',
        # token_pattern=r'\w{1,}',
        stop_words=stopWords,
        ngram_range=(1, 3),
        max_features=50000,
        norm='l2',
        min_df=3,
        max_df=0.6)

    dat['description'] = dat['description'].fillna(' ')
    dat['title'] = dat['title'].fillna(' ')
    tfidf.fit(dat['description'])
    tfidf_title.fit(dat['title'])

    dat_des_tfidf = tfidf.transform(dat['description'])
    dat_title_tfidf = tfidf.transform(dat['title'])

    # Get Key Components for tfidf matrix
    print('NLP - svd')
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf.transform(dat['description']))

    print(svd_obj.explained_variance_ratio_)

    svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_title.fit(tfidf.transform(dat['title']))

    print(svd_title.explained_variance_ratio_)

    dat_svd = pd.DataFrame(svd_obj.transform(dat_des_tfidf))
    dat_svd.columns = ['svd_des_' + str(i + 1) for i in range(n_comp)]
    # dat = pd.concat([dat, dat_svd], axis=1)
    dat = dat.join(dat_svd)

    dat_title_svd = pd.DataFrame(svd_title.transform(dat_title_tfidf))
    dat_title_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    # dat = pd.concat([dat, dat_title_svd], axis=1)
    dat = dat.join(dat_title_svd)

    return dat



def translate_russian_category(dat):
    parent_category_name_map = {"Личные вещи": "Personal belongings",
                                "Для дома и дачи": "For the home and garden",
                                "Бытовая электроника": "Consumer electronics",
                                "Недвижимость": "Real estate",
                                "Хобби и отдых": "Hobbies & leisure",
                                "Транспорт": "Transport",
                                "Услуги": "Services",
                                "Животные": "Animals",
                                "Для бизнеса": "For business"}

    region_map = {"Свердловская область": "Sverdlovsk oblast",
                  "Самарская область": "Samara oblast",
                  "Ростовская область": "Rostov oblast",
                  "Татарстан": "Tatarstan",
                  "Волгоградская область": "Volgograd oblast",
                  "Нижегородская область": "Nizhny Novgorod oblast",
                  "Пермский край": "Perm Krai",
                  "Оренбургская область": "Orenburg oblast",
                  "Ханты-Мансийский АО": "Khanty-Mansi Autonomous Okrug",
                  "Тюменская область": "Tyumen oblast",
                  "Башкортостан": "Bashkortostan",
                  "Краснодарский край": "Krasnodar Krai",
                  "Новосибирская область": "Novosibirsk oblast",
                  "Омская область": "Omsk oblast",
                  "Белгородская область": "Belgorod oblast",
                  "Челябинская область": "Chelyabinsk oblast",
                  "Воронежская область": "Voronezh oblast",
                  "Кемеровская область": "Kemerovo oblast",
                  "Саратовская область": "Saratov oblast",
                  "Владимирская область": "Vladimir oblast",
                  "Калининградская область": "Kaliningrad oblast",
                  "Красноярский край": "Krasnoyarsk Krai",
                  "Ярославская область": "Yaroslavl oblast",
                  "Удмуртия": "Udmurtia",
                  "Алтайский край": "Altai Krai",
                  "Иркутская область": "Irkutsk oblast",
                  "Ставропольский край": "Stavropol Krai",
                  "Тульская область": "Tula oblast"}

    category_map = {"Одежда, обувь, аксессуары": "Clothing, shoes, accessories",
                    "Детская одежда и обувь": "Children's clothing and shoes",
                    "Товары для детей и игрушки": "Children's products and toys",
                    "Квартиры": "Apartments",
                    "Телефоны": "Phones",
                    "Мебель и интерьер": "Furniture and interior",
                    "Предложение услуг": "Offer services",
                    "Автомобили": "Cars",
                    "Ремонт и строительство": "Repair and construction",
                    "Бытовая техника": "Appliances",
                    "Товары для компьютера": "Products for computer",
                    "Дома, дачи, коттеджи": "Houses, villas, cottages",
                    "Красота и здоровье": "Health and beauty",
                    "Аудио и видео": "Audio and video",
                    "Спорт и отдых": "Sports and recreation",
                    "Коллекционирование": "Collecting",
                    "Оборудование для бизнеса": "Equipment for business",
                    "Земельные участки": "Land",
                    "Часы и украшения": "Watches and jewelry",
                    "Книги и журналы": "Books and magazines",
                    "Собаки": "Dogs",
                    "Игры, приставки и программы": "Games, consoles and software",
                    "Другие животные": "Other animals",
                    "Велосипеды": "Bikes",
                    "Ноутбуки": "Laptops",
                    "Кошки": "Cats",
                    "Грузовики и спецтехника": "Trucks and buses",
                    "Посуда и товары для кухни": "Tableware and goods for kitchen",
                    "Растения": "Plants",
                    "Планшеты и электронные книги": "Tablets and e-books",
                    "Товары для животных": "Pet products",
                    "Комнаты": "Room",
                    "Фототехника": "Photo",
                    "Коммерческая недвижимость": "Commercial property",
                    "Гаражи и машиноместа": "Garages and Parking spaces",
                    "Музыкальные инструменты": "Musical instruments",
                    "Оргтехника и расходники": "Office equipment and consumables",
                    "Птицы": "Birds",
                    "Продукты питания": "Food",
                    "Мотоциклы и мототехника": "Motorcycles and bikes",
                    "Настольные компьютеры": "Desktop computers",
                    "Аквариум": "Aquarium",
                    "Охота и рыбалка": "Hunting and fishing",
                    "Билеты и путешествия": "Tickets and travel",
                    "Водный транспорт": "Water transport",
                    "Готовый бизнес": "Ready business",
                    "Недвижимость за рубежом": "Property abroad"}

    dat['region'] = dat['region'].apply(lambda x: region_map[x])
    dat['parent_category_name'] = dat['parent_category_name'].apply(lambda x: parent_category_name_map[x])
    dat['category_name'] = dat['category_name'].apply(lambda x: category_map[x])

    return dat
