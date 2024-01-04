# 使い方
- requirements.txtを使い、pipで必要なものをすべて入れてください
- ルートフォルダ内にconfigフォルダを作り、その中に"setup.json"を作成してください。
- setup.jsonの中身は以下の通りです。環境に合わせてパスは書き換えてください。

 {
    "path_train": "/Users/hogehoge/",
    "path_test": "/Users/hogehoge/",
    "path_result": "/Users/hogehoge/",
    "pattern_red": "*.JPEG", 
    "root_folder": "/Users/hogehoge/PnP-PDS/"
}
  - path_train デノイザーをトレーニングするときに使います。画像復元するだけなら適当で良いです。
  - path_test 復元したい画像が入っているフォルダです
  - path_restult 復元結果が格納されるフォルダです
  - pattern_red JPEGなら.JPEGと書いてください。その他も同様です。大文字小文字が区別されます。
  - root_folder フォルダを置いている場所を指定してください
 
- とりあえず、test.pyからeval_restorationを呼び出してみてください。
 - gaussian_nl ガウシアンノイズの標準偏差です。加える場合は、0.005〜0.02とかがちょうどいいです
 - sp_nl スパースノイズが重畳する割合です。0〜1の範囲です。
 - poisson_alpha　ポアソンノイズのスケーリング係数です。100ぐらいがちょうどいいです
 - gamma1, gamma2 PnP-PDSのステップサイズです
 - alpha_n 制約付き画像復元問題における、データ制約項の係数です。理論的には1がちょうどいいですが、0.9ぐらいにしたほうが良い結果がでます。
 - alpha_s 制約付き画像復元問題における、スパースノイズ項の係数です。alpha_nと同様に、0.9ぐらいが良いです。
 - myLambda additiveなformulationにおける、データ項の係数です。大きくするとデータ項を重視して、小さくすると正則化項を重視します。適切な値はタスクによって違うので、手探りで見つけるしかないです。
 - architecture デノイザーの名前です
 - deg_op "blur", "random_sampling", "Id"の3種類を指定できます。論文中のPhiです
 - method 復元手法です。詳しくはpds.pyを覗いてみてください。ポアソンノイズの場合は、ours-CかcomparisonC-1, comparisonC-2, comparisonC-3のどれかを指定することになると思います。ガウシアンノイズだけのときはours-AとかcomparisonA-X, ガウシアン＋スパースのときはours-BとかcomparisonB-Xを指定します。
 - m1, m2, gammaInADMMStep1 ADMMで使うパラメータです。詳しくはpds.pyを見れば分かると思います。

出来合いのコードなので、うまく動かないこともあると思います。いつでもLINE等ください。
