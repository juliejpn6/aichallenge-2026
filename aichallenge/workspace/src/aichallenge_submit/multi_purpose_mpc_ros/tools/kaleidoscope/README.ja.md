# Kaleidoscope

日本語 | [English](README.md)

Kaleidoscopeは、`multi_purpose_mpc_ros`で使用しているPython/Tk製の
オフラインtrajectory editorです。将来このツールだけを独立して公開できるよう、
MPCの実行時コードから分離してこのディレクトリに配置しています。

対応環境は、運営提供のAutomotive AI Challenge用Dockerイメージです。
ホストのPython環境からの直接実行はサポートしません。このeditorはROS nodeではなく、
Docker内のPython/Tkアプリとして起動します。

## 利用前提となるリポジトリ構成

Kaleidoscopeは、本AI Challengeリポジトリのディレクトリ構成を変更せずに使用して
ください。このリポジトリは、必ず次の位置へcloneまたは配置します。

```text
aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/tools/kaleidoscope/
```

利用者は、`multi_purpose_mpc_ros`、その配下の`env/`、同階層にある
`aichallenge_submit_launch`、およびmap・trajectoryファイルの配置や名前を変更しない
ことを前提とします。この構成を変更した環境はサポート対象外です。

## 起動方法

ホスト側でAI Challengeリポジトリのルートへ移動し、Autoware commandコンテナへ
入ります。

```bash
cd /path/to/aichallenge-2025
make autoware-bash
```

コンテナ内でKaleidoscopeのディレクトリへ移動して起動します。

```bash
cd /aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/tools/kaleidoscope
python3 -m kaleidoscope
```

引数を省略すると、リポジトリ内のMPC用trajectoryとLanelet2 mapを自動検出し、
周回経路として開きます。

任意のファイルを開く場合は、コンテナ内でパスを指定します。

```bash
python3 -m kaleidoscope \
  --trajectory ../../env/final_ver3/traj_mincurv.csv \
  --osm ../../../aichallenge_submit_launch/map/lanelet2_map.osm \
  --circular
```

指定できる引数は次のとおりです。

```text
--trajectory <trajectory CSV>
--osm <Lanelet2 OSM>
--circular                  周回経路
--open                      非周回経路
```

ホスト側でのvenv作成や`pip install`は不要です。ホスト側から
`python3 -m kaleidoscope`を実行しないでください。

## GUIが表示されない場合

ホスト側で次を確認します。

```bash
export XAUTHORITY=~/.Xauthority
./setup.bash doctor
```

`DISPLAY`または`XAUTHORITY`に警告が出る場合は、`setup.bash doctor`の案内に従って
X11設定を修正してから、もう一度`make autoware-bash`でコンテナへ入ります。

## 入力ファイル

通常の編集には次のファイルを使用します。

- trajectory CSV
- 表示用のLanelet2 OSM

壁とのclearance検証を使用する場合は、追加で次のファイルが必要です。

- occupancy-grid YAML
- YAMLから参照されるPGM画像

既定構成では、trajectory CSVと同じディレクトリにある
`occupancy_grid_map.yaml`を探索します。

## 基本的な編集手順

1. `Open Traj`で編集対象のCSVを開きます。
2. `Circular`の設定が経路に合っていることを確認します。
3. `Validate`で入力データを検証します。
4. 点を編集し、必要に応じて`Normalize Geometry`を実行します。
5. 速度情報を再生成する場合は`Recompute Speed`を実行します。
6. Before/Candidateの比較結果を確認してcandidateを適用します。
7. 再度`Validate`を実行します。
8. 最初は`Save As`で別名保存します。

元ファイルの誤上書きを避けるため、編集結果は
`*_edited.csv`、`*_normalized.csv`、`*_speed_profiled.csv`などの名前で保存することを
推奨します。

## 実行時依存

以下は運営Docker内に必要な依存関係です。利用者がホストへ個別にインストールする
必要はありません。

- Python 3.10以降
- Tkinter（Ubuntuでは`python3-tk`）
- defusedxml
- PyYAML

editor GUIは`rclpy`、ROS topic、ROS service、NumPy、pandas、OpenCVには依存して
いません。

## 現段階の注意事項

- 既定のtrajectory、map、車体寸法、余白は、このリポジトリの現行構成に基づく
  ローカル設定です。Automotive AI Challenge 2026の公式確定仕様ではありません。
- editorのclearance判定が`SAFE`でも、AWSIMや実車での非接触を保証しません。
- 保存した`vx_mps`と`ax_mps2`はオフラインCSVのmetadataです。現行MPCでは実行時の
  速度上限処理が優先されます。
- map・trajectoryを独立GitHubリポジトリへ同梱する場合は、それぞれの再配布条件を
  確認して明記する必要があります。

## ライセンス

Kaleidoscopeは[Apache License 2.0](LICENSE)で提供します。
