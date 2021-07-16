import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from contextlib import contextmanager
from time import time


path ='./data/train'
allFiles = glob.glob(path + "/*.csv") # 指定したフォルダーの全エクセルファイルを変数に代入します


frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0) # エクセルをデータフレームとして読み込む
    list_.append(df)
frame = pd.concat(list_, join='inner') # joinをinnerに指定

train_df = frame
test_df = pd.read_csv('./data/test.csv')

train_df=train_df.replace('5000㎡以上',5000)
test_df=test_df.replace('5000㎡以上',5000)

train_df=train_df.replace('2000㎡以上',2000)
test_df=test_df.replace('2000㎡以上',2000)

train_df=train_df.replace('30分?60分',45)
test_df=test_df.replace('30分?60分',45)

train_df=train_df.replace('2H?',120)
test_df=test_df.replace('2H?',120)

train_df=train_df.replace('1H30?2H',105)
test_df=test_df.replace('1H30?2H',105)

train_df=train_df.replace('2H?',120)
test_df=test_df.replace('2H?',120)

train_df=train_df.replace('1H?1H30',75)
test_df=test_df.replace('1H?1H30',75)


y_list = {}
for i in train_df["建築年"].value_counts().keys():
   if "平成" in i:
      num = float(i.split("平成")[1].split("年")[0])
      year = 33 - num
   if "令和" in i:
      num = float(i.split("令和")[1].split("年")[0])
      year = 3 - num
   if "昭和" in i:
      num = float(i.split("昭和")[1].split("年")[0])
      year = 96 - num
   y_list[i] = year
y_list["戦前"] = 76
train_df["建築年"] = train_df["建築年"].replace(y_list)

y_list = {}
for i in test_df["建築年"].value_counts().keys():
   if "平成" in i:
      num = float(i.split("平成")[1].split("年")[0])
      year = 33 - num
   if "令和" in i:
      num = float(i.split("令和")[1].split("年")[0])
      year = 3 - num
   if "昭和" in i:
      num = float(i.split("昭和")[1].split("年")[0])
      year = 96 - num
   y_list[i] = year
y_list["戦前"] = 76
test_df["建築年"] = test_df["建築年"].replace(y_list)


year = {
        "年第１四半期": ".25",
        "年第２四半期": ".50",
        "年第３四半期": ".75",
        "年第４四半期": ".99"
}
year_list = {}
for i in train_df["取引時点"].value_counts().keys():
   for k, j in year.items():
      if k in i:
         year_rep = i.replace(k, j)
   year_list[i] = year_rep
train_df["取引時点"] = train_df["取引時点"].replace(year_list).astype(float)


for i in test_df["取引時点"].value_counts().keys():
   for k, j in year.items():
      if k in i:
         year_rep = i.replace(k, j)
   year_list[i] = year_rep
test_df["取引時点"] = test_df["取引時点"].replace(year_list).astype(float)

#y_max=9.0
#idx_use = train_df['取引価格（総額）_log'] < y_max
#train_df['取引価格（総額）_log'] = np.where(idx_use, train_df['取引価格（総額）_log'], y_max)
train_df['地域']=train_df['建築年']-train_df['取引時点']
test_df['地域']=test_df['建築年']-test_df['取引時点']
#train_df['間口']=train_df['市区町村名'].str.split('市', expand=True)
#test_df['間口']=test_df['市区町村名'].str.split('市', expand=True)

def create_one_hot_encoding(input_df):
    use_columns = [
     #'用途',
    ]
    out_df = pd.DataFrame()
    for column in use_columns:

        # あまり巨大な行列にならないよう, 出現回数が 20 回を下回るカテゴリは考慮しない
        vc = train_df[column].value_counts()
        vc = vc[vc > 10]

        # 明示的に catgories を指定して, input_df によらず列の大きさが等しくなるようにする
        cat = pd.Categorical(input_df[column], categories=vc.index)

        # このタイミングで one-hot 化
        out_i = pd.get_dummies(cat)
        # column が Catgory 型として認識されているので list にして解除する (こうしないと concat でエラーになる)
        out_i.columns = out_i.columns.tolist()
        out_i = out_i.add_prefix(f'{column}=')
        out_df = pd.concat([out_df, out_i], axis=1)
    return out_df


def create_numeric_feature(input_df):
    use_columns = [
         '建ぺい率（％）','地域',
         '面積（㎡）',"建築年","取引時点",
         '容積率（％）','最寄駅：距離（分）'
           ]

    return input_df[use_columns].copy()

def create_string_length_feature(input_df):
    out_df = pd.DataFrame()

    str_columns = [
        # and more
    ]

    for c in str_columns:
        out_df[c] = input_df[c].str.len()

    return out_df.add_prefix('StringLength__')

def create_count_encoding_feature(input_df):
    use_columns = [
      '都道府県名', '市区町村名', '地区名', '最寄駅：名称',
       '間取り','都市計画',
       '今後の利用目的', '取引の事情等','建物の構造','市区町村コード', '用途', '改装',
    ]

    out_df = pd.DataFrame()
    for column in use_columns:
        vc = train_df[column].value_counts()
        out_df[column] = input_df[column].map(vc)

    return out_df.add_prefix('CE_')

from contextlib import contextmanager
from time import time

class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


from tqdm import tqdm

def to_feature(input_df):
    """input_df を特徴量行列に変換した新しいデータフレームを返す.
    """

    processors = [
        create_numeric_feature,
        #create_string_length_feature,
        create_count_encoding_feature,
        #create_one_hot_encoding
    ]

    out_df = pd.DataFrame()

    for func in tqdm(processors, total=len(processors)):
        with Timer(prefix='create' + func.__name__ + ' '):
            _df = func(input_df)

        # 長さが等しいことをチェック (ずれている場合, func の実装がおかしい)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)

    return out_df


train_feat_df = to_feature(train_df)

test_feat_df = to_feature(test_df)

print(train_feat_df.shape)
import category_encoders as ce

use_columns = [
       '都道府県名', '市区町村名', '地区名', '最寄駅：名称',
       '間取り','都市計画','間口',
       '今後の利用目的', '取引の事情等','建物の構造','市区町村コード', '用途', '改装',
        ]

out_df = pd.DataFrame()
target_col= '取引価格（総額）_log'
for cate_col in use_columns:
    te = ce.TargetEncoder(cols=cate_col)
    te_train = te.fit_transform(train_df[cate_col], train_df[target_col])
    te_test = te.transform(test_df[cate_col])
    train_feat_df = pd.concat([train_feat_df, te_train], axis=1)
    test_feat_df = pd.concat([test_feat_df, te_test], axis=1)

    oe = ce.OrdinalEncoder(cols=cate_col, drop_invariant=True)
    oe_train_df = oe.fit_transform(train_df[cate_col])
    oe_test_df = oe.fit_transform(test_df[cate_col])
    train_feat_df = pd.concat([train_feat_df, oe_train_df], axis=1)
    test_feat_df = pd.concat([test_feat_df, oe_test_df], axis=1)

    jse = ce.JamesSteinEncoder(cols=cate_col, drop_invariant=True)
    jse_train_df = jse.fit_transform(train_df[cate_col], train_df[target_col])
    jse_test_df = jse.transform(test_df[cate_col])
    train_feat_df = pd.concat([train_feat_df, jse_train_df], axis=1)
    test_feat_df = pd.concat([test_feat_df, jse_test_df], axis=1)


    cbe = ce.CatBoostEncoder(cols=cate_col, random_state=42)
    cbe_train_df = cbe.fit_transform(train_df[cate_col], train_df[target_col])
    cbe_test_df = cbe.transform(test_df[cate_col])
    train_feat_df = pd.concat([train_feat_df, cbe_train_df], axis=1)
    test_feat_df = pd.concat([test_feat_df, cbe_test_df], axis=1)


print(train_feat_df.shape)

params = {
    # 目的関数. これの意味で最小となるようなパラメータを探します.
    'objective': 'MAE',

    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'feature_fraction': 0.45199999999999996,
    'feature_pre_filter': False,
    'metric': 'l1',
    'min_child_samples': 10,
    'num_leaves': 31,

     # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
    # がそれだけ木を作るため学習に時間がかかります
    'learning_rate': 0.01,

    # L2 Reguralization
    'reg_lambda': 1.5553820454973332e-08,
    # こちらは L1
    'reg_alpha': 0.31611690467988246,

    # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'max_depth': 5,

    # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    'n_estimators': 300000,

    # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    'colsample_bytree': .6,

    # bagging の頻度と割合
    'subsample_freq': 3,
    'subsample': .9,

    # 特徴重要度計算のロジック(後述)
    'importance_type': 'gain',
    'random_state': 71,
}


def sample_scheduler_func(current_lr, eval_history, best_round, is_higher_better):
    """次のラウンドで用いる学習率を決定するための関数 (この中身を好きに改造する)

    :param current_lr: 現在の学習率 (指定されていない場合の初期値は None)
    :param eval_history: 検証用データに対する評価指標の履歴
    :param best_round: 現状で最も評価指標の良かったラウンド数
    :param is_higher_better: 高い方が性能指標として優れているか否か
    :return: 次のラウンドで用いる学習率

    NOTE: 学習を打ち切りたいときには callback.EarlyStopException を上げる
    """
    # 学習率が設定されていない場合のデフォルト
    current_lr = current_lr or 0.05

    # 試しに 20 ラウンド毎に学習率を半分にしてみる
    if len(eval_history) % 20 == 0:
        current_lr /= 2

    # 小さすぎるとほとんど学習が進まないので下限も用意する
    min_threshold = 0.001
    current_lr = max(min_threshold, current_lr)

    return current_lr

class LrSchedulingCallback(object):
    """ラウンドごとの学習率を動的に制御するためのコールバック"""

    def __init__(self, strategy_func):
        # 学習率を決定するための関数
        self.scheduler_func = strategy_func
        # 検証用データに対する評価指標の履歴
        self.eval_metric_history = []

    def __call__(self, env):
        # 現在の学習率を取得する
        current_lr = env.params.get('learning_rate')

        # 検証用データに対する評価結果を取り出す (先頭の評価指標)
        first_eval_result = env.evaluation_result_list[0]
        # スコア
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているか否か
        is_higher_better = first_eval_result[3]

        # 評価指標の履歴を更新する
        self.eval_metric_history.append(metric_score)
        # 現状で最も優れたラウンド数を計算する
        best_round_find_func = np.argmax if is_higher_better else np.argmin
        best_round = best_round_find_func(self.eval_metric_history)

        # 新しい学習率を計算する
        new_lr = self.scheduler_func(current_lr=current_lr,
                                     eval_history=self.eval_metric_history,
                                     best_round=best_round,
                                     is_higher_better=is_higher_better)

        # 次のラウンドで使う学習率を更新する
        update_params = {
            'learning_rate': new_lr,
        }
        env.model.reset_parameter(update_params)
        env.params.update(update_params)

    @property
    def before_iteration(self):
        # コールバックは各イテレーションの後に実行する
        return False







import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from typing import Union, Optional, Iterable
from sklearn.metrics import accuracy_score # モデル評価用(正答率)
from sklearn.metrics import mean_absolute_error

def fit_lgbm(X,
             y,
             cv,
             params: dict=None,
             verbose: int=50):
    """lightGBM を CrossValidation の枠組みで学習を行なう function"""

    # パラメータがないときは、空の dict で置き換える
    if params is None:
        params = {}


    lr_scheduler_cb = LrSchedulingCallback(strategy_func=sample_scheduler_func)
    callbacks = [
        lr_scheduler_cb,
    ]

    models = []
    # training data の target と同じだけのゼロ配列を用意
    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv):
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)

        with Timer(prefix='fit fold={} '.format(i)):
            clf.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=10000,
                    verbose=verbose,
                    callbacks=callbacks,
                    )

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        print(f'Fold {i} MAE: {mean_absolute_error(y_valid, pred_i) ** .5:.4f}')

    score = mean_absolute_error(y, oof_pred) ** .5
    print('-' * 50)
    print('FINISHED | Whole MAE: {:.4f}'.format(score))
    return oof_pred, models


from sklearn.model_selection import StratifiedKFold
from typing import Union, Tuple
#https://www.guruguru.science/competitions/16/discussions/092c2925-6a63-4e65-8057-6ea50fc660dd/
class ContinuousStratifiedKFold:
    def __init__(self, n_split: int=5, shuffle: bool=True, random_state: int=42) -> None:
        self.n_split = n_split
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: pd.DataFrame, reference: pd.Series, bins: int=10,
                      suffle: Union[bool]=None, random_state: Union[int]=None) -> Tuple[np.ndarray, np.ndarray]:

        shuffle = shuffle if suffle is not None else self.shuffle
        random_state = random_state if random_state is not None else self.random_state
        min_ref, max_ref = int(reference.min() - 1), int(reference.max() + 1)
        cut_threshold = np.linspace(min_ref, max_ref, bins)
        out = pd.cut(reference, bins=cut_threshold, labels=False)

        skf = StratifiedKFold(self.n_split, shuffle=shuffle, random_state=random_state)
        for train_idx, val_idx in skf.split(X, out):
            yield train_idx, val_idx

y = train_df['取引価格（総額）_log'].values
y = np.log1p(y)

fold = ContinuousStratifiedKFold(n_split=5)
cv = list(fold.split(train_feat_df, y))

oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params, verbose=500)

def revert_to_real(y_log):
    _pred = np.expm1(y_log)
    _pred = np.where(_pred < 0, 0, _pred)
    return _pred

pred = np.array([model.predict(test_feat_df.values) for model in models])
pred = np.mean(pred, axis=0)
pred = revert_to_real(pred)
sub_df = pd.DataFrame({ '取引価格（総額）_log': pred })

sub_df.to_excel('0716__submission_lgbm_1.xlsx', index=False)

