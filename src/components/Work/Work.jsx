import React from 'react';
import './work.scss';
export const Work = () => {
  const [add, setAdd] = React.useState(1);

  return (
    <div className="container">
      <div className="read">
        <div className="readLeft">
          <h2 className="read__title">Мазмұны</h2>
          <ul className="read__menu">
            <li onClick={() => setAdd(1)} className="read__item">
              <h3 className="read__menu-title1">№1 практикалық тапсырма. Түс датчигі</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(2)}>
              <h3 className="read__menu-title1">№2 практикалық тапсырма. Бағдаршам-робот</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(3)}>
              <h3 className="read__menu-title1">№3 практикалық тапсырма. Ультрадыбыс датчигі</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(4)}>
              <h3 className="read__menu-title1">
                №4 практикалық тапсырма. Робот түрлері және оларды қолдану
              </h3>
            </li>
            <li className="read__item" onClick={() => setAdd(5)}>
              <h3 className="read__menu-title1">№5 практикалық тапсырма. Гироскопиялық датчик</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(6)}>
              <h3 className="read__menu-title1">№6 практикалық тапсырма. Бүрылыстар</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(7)}>
              <h3 className="read__menu-title1">
                №7 практикалық тапсырма. Роботтардың қолданылу саласы
              </h3>
            </li>
            <li className="read__item" onClick={() => setAdd(8)}>
              <h3 className="read__menu-title1">
                №8 практикалық тапсырма. Роботтың сызық бойымен қозғалысы
              </h3>
            </li>
            <li className="read__item" onClick={() => setAdd(9)}>
              <h3 className="read__menu-title1">№9 практикалық тапсырма. Робо-сумо</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(10)}>
              <h3 className="read__menu-title1">№10 практикалық тапсырма. Датчиктер</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(11)}>
              <h3 className="read__menu-title1">№11 практикалық тапсырма. Программа блоктары</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(12)}>
              <h3 className="read__menu-title1">№12 практикалық тапсырма. Циклдік алгоритмдер</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(13)}>
              <h3 className="read__menu-title1">№13 практикалық тапсырма. Тармақталу</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(14)}>
              <h3 className="read__menu-title1">№14 зертханалық жұмыс. Роботтың алгоритмін құру</h3>
            </li>
            <li className="read__item" onClick={() => setAdd(15)}>
              <h3 className="read__menu-title1 read__menu-title2">
                №15 зертханалық жұмыс. Робо-сумо программасын құру
              </h3>
            </li>
          </ul>
        </div>
        <div className="readRight">
          {add === 1 && (
            <>
              <h2 className="readRight__title">
                №1 практикалық тапсырма: Деректерді Алдын Ала Өңдеу
              </h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Тапсырма:</b> "Титаник" деректер жинағындағы жетіспейтін мәндерді толтырып,
                  сандық емес айнымалыларды сандыққа айналдырыңыз.
                </p>
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <p>import pandas as pd</p>
                <br />
                <p># Деректерді жүктеу</p>
                <p>data =</p>
                <p>
                  pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
                </p>
                <br />
                <p># Жетпейтін мәндерді толтыру</p>
                <p>data['Age'].fillna(data['Age'].median(), inplace=True)</p>
                <p>data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)</p> <br />
                <p># Сандық емес айнымалыларды кодтау</p>
                <p>data = pd.get_dummies(data, columns=['Sex', 'Embarked'])</p>
                <br />
                <p># Қажетсіз бағандарды жою</p>
                <p>data = data.drop(columns=['Name', 'Ticket', 'Cabin'])</p>
                <br />
                <p># Нәтижені көрсету</p>
                <p>print(data.head())</p>
              </p>
            </>
          )}{' '}
          {add === 2 && (
            <>
              <h2 className="readRight__title">
                №2 практикалық тапсырма: Деректерді Визуализациялау
              </h2>
              <p className="readRight__descr">
                <p>
                  <b>Тапсырма: </b> "Ирис" деректер жинағының жұптық диаграммасын құрыңыз.
                </p>
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <p>import seaborn as sns</p>
                <p>import matplotlib.pyplot as plt</p>
                <p>import pandas as pd</p> <br />
                <p># Деректерді жүктеу</p>
                <p>data =</p>
                <p>
                  pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
                </p>
                <p># Жұптық диаграмма</p>
                <p>sns.pairplot(data, hue='species')</p>
                <p>plt.show()</p>
              </p>
            </>
          )}{' '}
          {add === 3 && (
            <>
              <h2 className="readRight__title">№3 практикалық тапсырма. Ультрадыбыс датчигі</h2>
              <p className="readRight__descr">
                <p>
                  <b>Тапсырма: </b> "Бостон тұрғын үйлері" деректерімен сызықтық регрессия моделін
                  құрыңыз.
                </p>{' '}
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <p>import pandas as pd</p>
                <p>from sklearn.model_selection import train_test_split</p>
                <p>from sklearn.linear_model import LinearRegression</p>
                <p>from sklearn.metrics import mean_squared_error, r2_score</p>
                <p>from sklearn.datasets import load_boston</p>
                <br />
                <p># Деректерді жүктеу</p>
                <p>boston = load_boston()</p>
                <p>X = pd.DataFrame(boston.data, columns=boston.feature_names)</p>
                <p>y = pd.Series(boston.target)</p> <br />
                <p># Деректерді бөлу</p>
                <p>
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                  random_state=42)
                </p>{' '}
                <br />
                <p># Модельді құру</p>
                <p>model = LinearRegression()</p>
                <p>model.fit(X_train, y_train)</p>
                <br />
                <p># Болжау және бағалау</p>
                <p>y_pred = model.predict(X_test)</p>
                <p>mse = mean_squared_error(y_test, y_pred)</p>
                <p>r2 = r2_score(y_test, y_pred)</p>
              </p>
            </>
          )}
          {add === 4 && (
            <>
              <h2 className="readRight__title">
                №4 практикалық тапсырма: Кластерлеу Әдістерін Қолдану
              </h2>
              <p className="readRight__descr">
                <p>
                  <b>Тапсырма:</b> "Ирис" деректер жинағын пайдаланып, K-Means кластерлеу жасаңыз.
                </p>{' '}
                <br />
                <p>
                  <b>Код:</b>
                </p>{' '}
                <br />
                <p>import pandas as pd</p>
                <p>from sklearn.cluster import KMeans</p>
                <p>import seaborn as sns</p>
                <p>import matplotlib.pyplot as plt</p>
                <br />
                <p># Деректерді жүктеу</p>
                <p>
                  data =
                  pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
                </p>
                <p>X = data.drop(columns=['species'])</p> <br />
                <p># K-Means кластерлеу</p>
                <p>kmeans = KMeans(n_clusters=3, random_state=42)</p>
                <p>data['cluster'] = kmeans.fit_predict(X)</p> <br />
                <p># Визуализация</p>
                <p>
                  sns.scatterplot(data=data, x='petal_length', y='petal_width', hue='cluster',
                  palette='viridis')
                </p>
                <p>plt.show()</p>
              </p>
            </>
          )}{' '}
          {add === 5 && (
            <>
              <h2 className="readRight__title">
                №5 практикалық тапсырма: Классификация Модельдерін Құру
              </h2>
              <p className="readRight__descr">
                <p>
                  <b>Тапсырма:</b> Decision Tree көмегімен "Ирис" деректер жинағы бойынша
                  классификация жасаңыз.
                </p>
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <br />
                <p>import pandas as pd</p>
                <p>from sklearn.model_selection import train_test_split</p>
                <p>from sklearn.tree import DecisionTreeClassifier</p>
                <p>from sklearn.metrics import accuracy_score</p> <br />
                <p># Деректерді жүктеу</p>
                <p>
                  data =
                  pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
                </p>
                <p>X = data.drop(columns=['species'])</p>
                <p>y = data['species']</p> <br />
                <p># Деректерді бөлу</p>
                <p>
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                  random_state=42)
                </p>{' '}
                <br />
                <p># Decision Tree моделін құру</p>
                <p>model = DecisionTreeClassifier(random_state=42)</p>
                <p>model.fit(X_train, y_train)</p> <br />
                <p># Болжау және бағалау</p>
                <p>y_pred = model.predict(X_test)</p>
                <p>accuracy = accuracy_score(y_test, y_pred)</p>
                <p>print(f'Дәлдік: {'{accuracy}'}')</p>
              </p>
            </>
          )}{' '}
          {add === 6 && (
            <>
              <h2 className="readRight__title">
                №6 практикалық тапсырма: Модельді Гиперпараметрлерін Оңтайландыру
              </h2>
              <p className="readRight__descr">
                <p>
                  <b>Тапсырма:</b> GridSearchCV көмегімен Random Forest моделінің гиперпараметрлерін
                  оңтайландыру.
                </p>
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <br />
                <p>import pandas as pd</p>
                <p>from sklearn.model_selection import train_test_split, GridSearchCV</p>
                <p>from sklearn.ensemble import RandomForestClassifier</p> <br />
                <p># Деректерді жүктеу</p>
                <p>
                  data =
                  pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
                </p>
                <p>X = data.drop(columns=['species'])</p>
                <p>y = data['species']</p>
                <br />
                <p># Деректерді бөлу</p>
                <p>
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                  random_state=42)
                </p>
                <br />
                <p># Random Forest моделін құру</p>
                <p>model = RandomForestClassifier(random_state=42)</p> <br />
                <p># Гиперпараметрлерді реттеу</p>
                <p>param_grid = {'{'}</p>
                <p> 'n_estimators': [10, 50, 100],</p>
                <p>'max_depth': [None, 10, 20, 30],</p>
                <p> 'min_samples_split': [2, 5, 10],</p>
                <p>{'}'}</p>
                <p>grid_search = GridSearchCV(model, param_grid, cv=5)</p>
                <p>grid_search.fit(X_train, y_train)</p>
                <p>print(f'Үздік параметрлер: {'{grid_search.best_params_}'}')</p>
              </p>
            </>
          )}
          {add === 7 && (
            <>
              <h2 className="readRight__title">
                №7 практикалық тапсырма: Нейрондық Желілерді Құру
              </h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Тапсырма:</b> Қарапайым нейрондық желіні MNIST деректер жинағында құрыңыз.
                </p>
                <br />
                <p>
                  <b>Код:</b>
                </p>
                <br />
                <p>import tensorflow as tf</p>
                <p>from tensorflow.keras.datasets import mnist</p>
                <p>from tensorflow.keras.models import Sequential</p>
                <p>from tensorflow.keras.layers import Dense, Flatten</p>
                <p>from tensorflow.keras.utils import to_categorical</p> <br />
                <p># Деректерді жүктеу</p>
                <p>(X_train, y_train), (X_test, y_test) = mnist.load_data()</p>
                <p>X_train, X_test = X_train / 255.0, X_test / 255.0</p>
                <p>y_train, y_test = to_categorical(y_train), to_categorical(y_test)</p> <br />
                <p># Нейрондық желіні құру</p>
                <p>model = Sequential([</p>
                <p> Flatten(input_shape=(28, 28)),</p>
                <p> Dense(128, activation='relu'),</p>
                <p> Dense(10, activation='softmax')</p>
                <p>])</p> <br />
                <p># Модельді компиляциялау</p>
                <p>
                  model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
                </p>{' '}
                <br />
                <p># Модельді оқыту</p>
                <p>model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))</p>
              </p>
            </>
          )}
          {add === 8 && (
            <>
              <h2 className="readRight__title">№8 практикалық тапсырма: Табиғи Тілді Өңдеу</h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Тапсырма:</b> Текстерді алдын ала өңдеп, сөздердің жиілік талдауын жасаңыз.
                </p>
                <br />
                <p>
                  <b> Код:</b>
                </p>
                <br />
                <p>import nltk</p>
                <p>from nltk.corpus import stopwords</p>
                <p>from nltk.tokenize import word_tokenize</p> <br />
                <p># Нүкте белгілері</p>
                <p>nltk.download('punkt')</p>
                <p>nltk.download('stopwords')</p>
                <br />
                <p># Мәтін</p>
                <p>
                  text = "Natural language processing (NLP) is a field of artificial intelligence
                  that enables{' '}
                </p>
                <p>computers to analyze and understand human language."</p> <br />
                <p># Токенизация</p>
                <p>tokens = word_tokenize(text.lower())</p> <br />
                <p># Стоп сөздер мен пунктуацияларды жою</p>
                <p>
                  filtered_tokens = [word for word in tokens if word.isalpha() and word not in
                  stopwords.words('english')]
                </p>{' '}
                <br />
                <p># Жиілік санақ</p>
                <p>freq = nltk.FreqDist(filtered_tokens)</p>
                <p>print(freq.most_common(5))</p>
              </p>
            </>
          )}
          {add === 9 && (
            <>
              <h2 className="readRight__title">
                №9 практикалық тапсырма: PCA арқылы Деректерді Қысқарту
              </h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Тапсырма: </b> "Ирис" деректер жинағында PCA қолдана отырып деректерді екі
                  өлшемге қысқартыңыз.
                </p>
                <br />
                <p>
                  <b> Код:</b>
                </p>
                <br />
                <p>import pandas as pd</p>
                <p>from sklearn.decomposition import PCA</p>
                <p>import seaborn as sns</p>
                <p>import matplotlib.pyplot as plt</p>
                <br />
                <p># Деректерді жүктеу</p>
                <br />
                <p>
                  data =
                  pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
                </p>
                <p>X = data</p>
              </p>
            </>
          )}
          {add === 10 && (
            <>
              <h2 className="readRight__title">№10 практикалық тапсырма: Нейрондық желілер</h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат: </b> Қарапайым нейрондық желіні құру және жаттықтыру.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> MNIST Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Деректерді жүктеу және қалыпқа келтіру.</li>
                    <li>2. Нейрондық желіні құру.</li>
                    <li>3. Нейрондық желіні жаттықтыру және бағалау.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
          {add === 11 && (
            <>
              <h2 className="readRight__title">
                №11 практикалық тапсырма: Модельдің гиперпараметрлерін оңтайландыру
              </h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат:</b> GridSearchCV көмегімен модельдің гиперпараметрлерін оңтайландыру.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> Iris Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Random Forest моделін құру.</li>
                    <li>2. GridSearchCV көмегімен гиперпараметрлерді оңтайландыру.</li>
                    <li>3. Нәтижелерді көрсету.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
          {add === 12 && (
            <>
              <h2 className="readRight__title">№12 практикалық тапсырма: Кластерлеу</h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат:</b> K-Means кластерлеу әдісін пайдаланып, деректерді кластерлеу.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> Iris Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Деректерді жүктеу және дайындау.</li>
                    <li>2. K-Means әдісін қолдану.</li>
                    <li>3. Нәтижелерді визуализациялау.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
          {add === 13 && (
            <>
              <h2 className="readRight__title">№13 практикалық тапсырма: Классификация</h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат:</b>Decision Tree классификациясын қолдану арқылы модель құру.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> Iris Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Деректерді жүктеу және бөлу.</li>
                    <li>2. Decision Tree моделін құру және жаттықтыру.</li>
                    <li>3. Нәтижелерді бағалау.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
          {add === 14 && (
            <>
              <h2 className="readRight__title">№14 практикалық тапсырма: Сызықтық регрессия</h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат:</b>Сызықтық регрессияны қолданып, деректерді модельдеу.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> Boston Housing Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Модельді құру және деректерді бөлу.</li>
                    <li>2. Сызықтық регрессияны қолдану.</li>
                    <li>3. Нәтижелерді бағалау.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
          {add === 15 && (
            <>
              <h2 className="readRight__title">
                №15 практикалық тапсырма: Деректерді алдын ала өңдеу
              </h2>
              <p className="readRight__descr">
                <br />
                <p>
                  <b>Мақсат:</b> Берілген деректер жинағын алдын ала өңдеуден өткізіп, негізгі
                  тазалау әдістерін қолдану.
                </p>
                <br />
                <p>
                  <b>Мәліметтер:</b> Titanic Dataset
                </p>
                <br />
                <p>
                  <b>Қадамдар:</b>
                </p>
                <p>
                  <ol>
                    <li>1. Деректерді жүктеп, зерттеу.</li>
                    <li>2. Пропущенные мәндерді толтыру.</li>
                    <li>3. Сандық емес айнымалыларды кодтау.</li>
                    <li>4. Қажетсіз бағандарды жою.</li>
                  </ol>
                </p>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
