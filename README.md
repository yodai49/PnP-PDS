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
 
- 
