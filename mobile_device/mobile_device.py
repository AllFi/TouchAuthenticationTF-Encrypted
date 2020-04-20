"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
from common import DataOwner, ModelOwner, LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(file_name, legal_user_id, gesture_type, random_state):
    features = ['X нач.', 'Y нач.', 'X уд.', 'Y уд.', 'X кон.', 'Y кон.',
                'Расстояние между нач. и уд.', 'Направление вектора между нач. и уд.',
                'Расстояние между нач. и кон.', 'Направление вектора между нач. и кон.',
                'Расстояние между уд. и кон.',
                'Длина траектории',
                'Расстояние от уд. до вектора жеста',
                'Среднее расстояние между точками', 'СКО расстояния между точками', 'Скорость нач.',
                'Скорость уд.', 'Скорость кон.', 'Средняя скорость', 'Время между нач. и кон.',
                'Время между уд. и кон.', 'Ускорение уд.',
                'Ускорение кон.', 'Размер нач.', 'Размер уд.', 'Размер кон.',
                'Средний размер',
                'Векторная X между нач. и уд.', 'Векторная Y между нач. и уд.',
                'Векторная Z между нач. и уд.',
                'Векторная X между нач. и кон.', 'Векторная Y между нач. и кон.',
                'Векторная Z между нач. и кон.', 'Угол между нач. и кон.',
                'Векторная X между уд. и кон.', 'Векторная Y между уд. и кон.',
                'Векторная Z между уд. и кон.', 'Угол между уд. и кон.',
                'Средняя векторная X', 'Средняя векторная Y']

    scaler = StandardScaler()
    touch_data = pd.read_csv('features.csv', index_col=None, sep=",", encoding="cp1251")
    touch_data = touch_data[touch_data["Тип жеста"] == gesture_type]

    # выбираем данные легитимного пользователя
    x_legal = touch_data[touch_data["Идентификатор пользователя"] == legal_user_id].loc[:, features]
    x_legal_size = x_legal.shape[0]
    x_legal_train_size = round(x_legal_size * 2 / 3)

    # перемешиваем и разделяем на обучающую и тестовую выборки
    x_legal = x_legal.sample(n=x_legal_size, random_state=random_state)
    x_legal_train = x_legal.iloc[:x_legal_train_size]
    x_legal_test = x_legal.iloc[x_legal_train_size:]

    # выбираем данные нелегитимных пользователей
    x_illegal = touch_data[touch_data["Идентификатор пользователя"] != legal_user_id].loc[:, features]

    # перемешиваем и разделяем на обучающую и тестовую выборки
    x_illegal = x_illegal.sample(n=x_legal_size, random_state=random_state)
    x_illegal_train = x_illegal.iloc[:x_legal_train_size]
    x_illegal_test = x_illegal.iloc[x_legal_train_size:]

    # объединяем обучающие выборки, стандартизируем, обучая при этом StandardScaler
    x_train = x_legal_train.append(x_illegal_train)
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled, columns=features)

    # объединяем тестовые выборки, стандартизируем
    x_test = x_legal_test.append(x_illegal_test)
    x_test_scaled = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test_scaled, columns=features)

    # размечаем выборки
    x_train["target"] = [1 if i < x_legal_train.shape[0] else 0 for i in range(x_train.shape[0])]
    x_test["target"] = [1 if i < x_legal_test.shape[0] else 0 for i in range(x_test.shape[0])]

    # перемешиваем выборки
    x_train = x_train.sample(n=x_train.shape[0], random_state=random_state )
    x_test = x_test.sample(n=x_test.shape[0], random_state=random_state)

    return x_train, x_test


def main(server):
    legal_user_id = 1
    gesture_type = 0
    random_state = 12345
    train_set, test_set = prepare_data("features.csv", legal_user_id, gesture_type, random_state)

    num_rows = train_set.shape[0]
    num_features = train_set.shape[1] - 1  # не считаем "target"
    num_epoch = 10
    batch_size = num_rows // 10
    num_batches = (num_rows // batch_size) * num_epoch

    model = LogisticRegression(num_features, random_state=random_state)
    model_owner = ModelOwner("mobile_device", train_set.columns[:-1])
    data_owner = DataOwner("mobile_device", train_set, test_set, batch_size)

    x_train, y_train = data_owner.provide_training_data()
    x_test, y_test = data_owner.provide_test_data()
    reveal_weights_op = model_owner.receive_weights(model.weights)

    with tfe.Session() as sess:
        sess.run([tfe.global_variables_initializer()], tag="init")
        model.fit(sess, x_train, y_train, num_batches)
        sess.run(reveal_weights_op, tag="reveal")
        model.evaluate(sess, x_test, y_test, data_owner)


def start_master(cluster_config_file=None):
    print("Starting mobile_device...")
    remote_config = tfe.RemoteConfig.load(cluster_config_file)
    tfe.set_config(remote_config)
    tfe.set_protocol(tfe.protocol.Pond())
    players = remote_config.players
    mobile_device = remote_config.server(players[0].name)
    main(mobile_device)


if __name__ == "__main__":
    start_master("config.json")