# trackdlo: クラス vs struct+フリー関数

## Hanの疑問と判断
### 最終判断


### なぜアルゴリズムをクラスで実装する必要があるのか？
trackdloのROS1のパッケージはアルゴリズムをクラスとして実装していた。そんなことをしなくても純粋な関数で書けば十分ではないかという疑問が浮かんだ。
trackdloは、現在と過去の状態を入力として、未来の状態を予測するアルゴリズムらしい。このときに過去の状態を保存する必要がある。
過去の状態を保存する必要があるので、クラスのメンバ変数として実装することでこれを満たしていた。

### なぜ
特にROS１ではクラスの強制がないのでtrackdloは前の状態を保持するためにクラスのメンバ変数で実現していた
ROS2ではクラスの形で書くことを矯正されるので、今度はtrackdloのアルゴリズムは純粋な関数として書くのが合理的である。

## なぜこの文書があるか

元の `trackdlo_ros2` パッケージでは trackdlo アルゴリズムがクラスとして実装されていた。
`pure_trackdlo` への移植にあたり、**クラスをやめて struct + フリー関数にする**設計判断を行った。
その根拠をここに記録する。

---

## 1. ROS1 における trackdlo クラスの由来

### イベント駆動型コールバックの制約

ROS1 の Subscriber はデータが届いたときだけコールバック関数を呼ぶ、イベント駆動型のモデルである。

```cpp
// ROS1 の典型的な起動
ros::Subscriber sub = nh.subscribe("/camera/pointcloud", 1, callback_function);
ros::spin();  // ここでコントロールをROSに渡す。以降は自分のコードは動かない
```

`ros::spin()` 以降は ROS がメインループを握る。データが来るたびに `callback_function` が呼ばれるが、
コールバック関数は「普通の関数」なので、**前回呼ばれたときのローカル変数は残っていない**。

しかし trackdlo アルゴリズムは本質的にフレーム間で状態を持ち越す必要がある:

| 変数 | 理由 |
|------|------|
| `Y` (ノード座標) | フレームN+1の初期値としてフレームNの結果を使う |
| `sigma2` (ガウス分散) | 収束後の値を次フレームの初期値にすると速く収束する |
| `correspondence_priors` | オクルージョン補間の結果を次フレームに引き継ぐ |
| `geodesic_coord` | 初期化時に一度だけ計算し、以降は再利用する |

### クラスによる解決

クラスのインスタンスは `main` が終わるまで生存する。
メンバ変数をコールバックをまたいだ「記憶」として使う。

```cpp
class TrackdloNode {
private:
    // フレーム間の状態をメンバ変数として保持
    Eigen::MatrixXd Y_;
    double sigma2_;
    std::vector<Eigen::MatrixXd> correspondence_priors_;
    std::vector<double> geodesic_coord_;

    ros::Subscriber sub_;

public:
    TrackdloNode() {
        Y_ = initialize_nodes();
        sigma2_ = 0.0;
        // コールバックに this を渡す → メンバ変数へアクセス可能になる
        sub_ = nh_.subscribe("/camera/pointcloud", 1,
                             &TrackdloNode::callback, this);
    }

    void callback(const sensor_msgs::PointCloud2& msg) {
        Eigen::MatrixXd X = convert(msg);
        // Y_ は前回のコールバックで更新済み
        tracking_step(X, Y_, sigma2_, correspondence_priors_, ...);
        // Y_ が更新される → 次のコールバックで使われる
    }
};

int main() {
    TrackdloNode node;  // コンストラクタで初期化・購読登録
    ros::spin();
}
```

### ROS1 はクラスを強制しない

`ros::NodeHandle` はグローバルスコープでも使えるため、
技術的にはグローバル変数 + 普通の関数でも動作する。
しかしコールバックパターン上、クラスのメンバ変数が最も自然な選択肢だったため、
**アルゴリズム本体までクラスに包まれることになった（ROSの都合に引きずられた設計）**。

---

## 2. ROS2 におけるアーキテクチャの変化

### `rclcpp::Node` 継承が事実上必須

ROS2 では、ノードは `rclcpp::Node` を継承したクラスとして書くのが標準イディオムである。

```cpp
// ROS2 の典型的な書き方
class MyNode : public rclcpp::Node {
public:
    MyNode() : Node("my_node") {
        // create_subscription は rclcpp::Node のメソッド
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/pointcloud", 10,
            std::bind(&MyNode::callback, this, std::placeholders::_1));
    }
private:
    void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) { ... }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());  // shared_ptr でラップして渡す
    rclcpp::shutdown();
}
```

`create_subscription` は `rclcpp::Node` のメソッドであり、`this->` 経由でしか呼べない。
`rclcpp::spin()` は `Node` の shared_ptr を要求する。
ROS1 のようにグローバル変数で逃げる手段がない。

### クラスの二重構造になる問題

ROS2 ラッパーはすでにクラスである。
そこに trackdlo クラスをメンバとして持つと「クラスの中にクラス」になる。

```cpp
// 問題のある設計
class TrackdloNode : public rclcpp::Node {
    trackdlo algo_;  // クラスの中にクラス

    void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        algo_.tracking_step(convert(msg));  // 状態がどこにあるか見えない
    }
};
```

`algo_` の内部状態（`Y_`, `sigma2_` 等）が隠蔽されており、何が変化しているか外から分からない。

### 合理的な分離: struct + フリー関数

ROS2 ラッパークラスが「フレーム間状態の保持」をすでに担っている。
アルゴリズム側にクラスは不要。struct + フリー関数にすれば接続がシンプルになる。

```cpp
// アルゴリズム側 (ROSに無関係)
struct TrackdloState {
    Eigen::MatrixXd Y;
    double sigma2 = 0.0;
    std::vector<Eigen::MatrixXd> correspondence_priors;
    std::vector<double> geodesic_coord;
};

struct TrackdloParams {
    double beta = 5.0;
    double lambda = 1.0;
    double mu = 0.05;
    // ...
};

void tracking_step(TrackdloState& state, const Eigen::MatrixXd& X,
                   const TrackdloParams& params, ...);

// ROS2 ラッパー側
class TrackdloNode : public rclcpp::Node {
    TrackdloState state_;   // ただの struct
    TrackdloParams params_; // ただの struct

    void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        tracking_step(state_, convert(msg), params_);  // ただの関数呼び出し
        // state_.Y が更新される → 次のコールバックで使われる
    }
};
```

状態の所在が明確で、アルゴリズムのテストも ROS なしで書ける。

---

## 3. ROS無しの場合

自分でループを書けば、ローカル変数がスコープ内に生き続けるのでクラスは不要。

```cpp
int main() {
    TrackdloState state;
    state.Y = initialize_nodes(...);
    state.sigma2 = 0.0;

    TrackdloParams params;
    params.beta = 5.0;

    while (true) {
        Eigen::MatrixXd X = get_pointcloud();
        tracking_step(state, X, params, ...);
        // state.Y は次のループでも生きている (スコープが main まで続く)
    }
}
```

---

## 4. まとめ

| 項目 | ROS1 | ROS2 | ROS無し |
|------|------|------|---------|
| ノードのクラス強制 | なし | あり（`rclcpp::Node` 継承） | — |
| フレーム間状態の管理場所 | trackdloクラスのメンバ変数 | ROS2ラッパークラスのメンバ変数 | `main` のローカル変数 |
| trackdloの理想形 | クラスでも動く（歴史的経緯） | struct + フリー関数 | struct + フリー関数 |

**結論**: ROS1 でのクラス実装はコールバック制約への対処であり、アルゴリズム自体の要求ではなかった。
ROS2 対応を見据えると、アルゴリズムを struct + フリー関数で実装しておくほうが、
ラッパーへの接続がシンプルになり、状態の所在も明確になる。
