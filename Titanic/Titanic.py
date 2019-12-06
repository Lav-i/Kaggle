import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_train_data = pd.read_csv('./Titanic/data/train.csv')
raw_test_data = pd.read_csv('./Titanic/data/test.csv')

raw_test_data['Survived'] = 0
combined_train_test = raw_train_data.append(raw_test_data)

combined_train_test.info()

# %%
# for Cabin
# 直接删除

combined_train_test = combined_train_test.drop(columns=['Cabin'])
combined_train_test.info()

# %%
# for Name
# 直接删除

combined_train_test = combined_train_test.drop(columns=['Name'])
combined_train_test.info()

# %%
# for PassengerId
# 直接删除

combined_train_test = combined_train_test.drop(columns=['PassengerId'])
combined_train_test.info()

# %%
# for Embarked
# 众数填充，one-hot编码

combined_train_test.Embarked.fillna(combined_train_test.Embarked.dropna().mode(), inplace=True)
combined_train_test = pd.concat([combined_train_test, pd.get_dummies(combined_train_test.pop('Embarked'))], axis=1)
combined_train_test.info()

# %%
# for Sex
# one-hot编码

combined_train_test = pd.concat([combined_train_test, pd.get_dummies(combined_train_test.pop('Sex'))], axis=1)
combined_train_test.info()

# %%
# for Fare
# 平均值填充，标准正态分布化

from sklearn import preprocessing

combined_train_test.Fare.fillna(combined_train_test.Fare.mean(), inplace=True)
scaler = preprocessing.StandardScaler()
combined_train_test['Fare'] = scaler.fit_transform(combined_train_test['Fare'].values.reshape((-1, 1)))
print(combined_train_test.head())

# %%
# for Age
# SVM预测填充，标准正态分布化

from sklearn import svm
from sklearn.metrics import median_absolute_error, log_loss
from sklearn.externals import joblib

age_df = combined_train_test[['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'C', 'Q', 'S', 'female', 'male']]
age_df.info()
age_df_notnull = age_df[(age_df['Age'].notnull())]
age_df_isnull = age_df[(age_df['Age'].isnull())]
X = age_df_notnull.values[:, 1:]
Y = age_df_notnull.values[:, 0]

try:
    age_model = joblib.load('./Titanic/age_model.pkl?')
except:
    age_model = svm.SVR(C=200, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                        gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
                        tol=0.001, verbose=False).fit(X[:700], Y[:700])
    print(median_absolute_error(Y[700:], age_model.predict(X[700:])))

joblib.dump(age_model, './Titanic/age_model.pkl')

combined_train_test.loc[combined_train_test['Age'].isnull(), ['Age']] = age_model.predict(age_df_isnull.values[:, 1:])

combined_train_test['Age'] = scaler.fit_transform(combined_train_test['Age'].values.reshape((-1, 1)))
print(combined_train_test.head())

# %%
# for Ticket
# 数字归一类，有字母的按照字母归类

combined_train_test['Ticket'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket'] = combined_train_test['Ticket'].apply(lambda x: 'U0' if x.isnumeric() else x)
combined_train_test['Ticket'] = pd.factorize(combined_train_test['Ticket'])[0]

print(combined_train_test.head())

# %%
# for family
# 增加家庭信息

combined_train_test['FamilySize'] = combined_train_test.Parch + combined_train_test.SibSp
print(combined_train_test.head())

# %%
# 保存文件
combined_train_test.to_csv('./Titanic/data/tmp.csv', index=False)

# %%
# 训练

from keras import Input
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

train_data = combined_train_test[:891]
test_data = combined_train_test[891:]

train_data_X = train_data.drop(columns=['Survived'])
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(columns=['Survived'])


def buildModel():
    input = Input(shape=(12,))
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    # model.summary()
    return model


try:
    model = load_model('./Titanic/model.h5')
except:
    model = buildModel()
    history = model.fit(train_data_X, train_data_Y,
                        validation_split=0.3,
                        batch_size=64, epochs=1000, verbose=2)
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(history.epoch, history.history['loss'], label='训练loss')
    plt.plot(history.epoch, history.history['val_loss'], label='验证loss')
    plt.plot(history.epoch, history.history['acc'], label='训练acc')
    plt.plot(history.epoch, history.history['val_acc'], label='验证acc')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

model.save('./Titanic/model.h5')

# %%
# 预测

h = (model.predict(test_data_X).reshape((-1,)) // 0.5).astype(np.int_)

result = pd.read_csv('./Titanic/data/gender_submission.csv')
result.Survived = h

result.to_csv('./Titanic/data/gender_submission.csv', index=False)
